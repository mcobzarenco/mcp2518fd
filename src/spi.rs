use core::fmt::Debug;

use bitfield::bitfield;
use embedded_can::{Id, StandardId};

#[cfg(not(feature = "async"))]
use embedded_hal::delay::DelayNs;
use embedded_hal::spi::Operation;
#[cfg(not(feature = "async"))]
use embedded_hal::spi::SpiDevice;

#[cfg(feature = "async")]
use embedded_hal_async::delay::DelayNs;
#[cfg(feature = "async")]
use embedded_hal_async::spi::SpiDevice;

use crate::memory::chip::{IoControlRegister, OscillatorControlRegister};
use crate::memory::controller::configuration::{
    CanControlRegister, DataBitTimeConfigurationRegister, NominalBitTimeConfigurationRegister,
    OperationMode, TimeBaseCounterRegister, TimeStampControlRegister,
    TransmitterDelayCompensationMode, TransmitterDelayCompensationRegister,
};
use crate::memory::controller::fifo::{
    FifoControlRegister, FifoNumber, FifoStatusRegister, TxEventFifoControlRegister,
    TxEventFifoStatusRegister, TxQueueControlRegister, TxQueueStatusRegister, UserAddressKind,
    UserAddressRegister,
};
use crate::memory::controller::filter::{
    FilterControlRegister, FilterNumber, FilterObjectRegister, MaskRegister,
};
use crate::memory::controller::interrupt::{
    InterruptCodeRegister, InterruptRegister, RxInterruptStatusRegister,
    RxOverflowInterruptStatusRegister, TxAttemptInterruptStatusRegister, TxInterruptStatusRegister,
};
use crate::memory::{is_valid_ram_address, ClearableFlags, Register, RepeatedRegister, SFRAddress};
use crate::message::rx::{RxHeader, RxMessage};
use crate::message::tx::{TxEventObject, TxHeader, TxMessage};
use crate::message::{len_for_dlc, MAX_FD_BUFFER_SIZE};
use crate::settings::{
    self, BitTimeConfiguration, FilterConfiguration, FilterMatchMode, RxFifoConfiguration,
    TxFifoConfiguration,
};
use crate::settings::{
    FifoConfiguration, IoConfiguration, OscillatorConfiguration, Pll, SysClkDivider,
    TxEventFifoConfiguration, TxQueueConfiguration,
};

/// Helper to round up SPI transfer sizes to align with 4-byte read/writes.
/// Only rounds up when needed (e.g., 1-4 rounds up to 4, 5-8 rounds up to 8).
fn round_up_spi_transfer_size(data_length: usize) -> usize {
    if data_length.is_multiple_of(4) {
        data_length
    } else {
        // Add the number of bytes needed to round up to the next multiple of 4
        data_length + (4 - data_length % 4)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum Error {
    /// Failed to read from the chip over SPI
    SPIRead,
    /// Failed to write to the chip over SPI
    SPIWrite,
    /// Attempted to access an invalid RAM address
    InvalidRamAddress(u16),
    /// Tried to read data from ram that was not a multiple of 4 bytes
    InvalidReadLength(usize),
    /// Tried to write data to ram that was not a multiple of 4 bytes
    InvalidWriteLength(usize),
    /// Tried to transmit a message through the TXQ, but the TXQ is not enabled
    TxQueueDisabled,
    /// Tried to transmit a message with a FIFO not configured for transmission
    FifoNotTx,
    /// Tried to send a message that was too big for the FIFO
    FifoTooSmall,
    /// FIFO is already full and can not take any more messages
    FifoFull,
    /// Tried to read a message from a FIFO not configured for reception
    FifoNotRx,
    /// The CRC from the chip did not match our calculated value for the data we received
    CrcMismatch,
    Other,
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum ConfigError {
    ChangeOpModeTimeout,
    ConfigurationModeTimeout,
    SPIFailedRAMEcho,
    PLLNotReady,
    Other(Error),
}

impl From<Error> for ConfigError {
    fn from(error: Error) -> Self {
        match error {
            Error::SPIRead | Error::SPIWrite => ConfigError::ConfigurationModeTimeout,
            _ => ConfigError::Other(error),
        }
    }
}

/// Number of times a CRC-protected read is repeated on a CRC mismatch before giving up.
const CRC_READ_RETRIES: usize = 3;

/// The `N` field of the RAM CRC instructions counts 32-bit words in a single byte, which caps
/// one CRC-protected RAM transfer at 255 words.
const RAM_CRC_MAX_TRANSFER_BYTES: usize = 255 * 4;

pub struct MCP2518FD<SPI> {
    spi: SPI,
    crc: crc::Crc<u16>,
    crc_protection: bool,
}

#[cfg_attr(not(feature = "async"), maybe_async::maybe_async)]
impl<SPI, SPIE> MCP2518FD<SPI>
where
    SPI: SpiDevice<u8, Error = SPIE>,
    SPIE: Debug,
{
    /// Constructs a new MCP2518FD controller from an SPI bus and CS GPIO pin
    pub fn new(spi: SPI) -> MCP2518FD<SPI> {
        Self {
            spi,
            // requires some somewhat expensive initialization, so keep a
            // referenc to the Crc instance
            crc: crc::Crc::<u16>::new(&crc::CRC_16_CMS),
            crc_protection: false,
        }
    }

    /// Enables or disables CRC-protected SPI transfers, see [`MCP2518FD::set_crc_protection`].
    pub fn with_crc_protection(mut self, enabled: bool) -> Self {
        self.crc_protection = enabled;
        self
    }

    /// Enables or disables CRC-protected SPI transfers (datasheet section 4.3).
    ///
    /// When enabled, every access uses the CRC variant of the SPI instruction: reads use
    /// `READ_CRC` and are retried up to three times on a mismatch, 32-bit register writes and
    /// RAM writes use `WRITE_CRC`, and single-byte register writes use `WRITE_SAFE`, which the
    /// device verifies before committing. A write CRC failure is not visible to the caller; the
    /// device reports it through `CRC.CRCERRIF` (and `CiINT.SPICRCIF` if enabled).
    ///
    /// The Linux `mcp251xfd` driver enables the equivalent for all supported chips. It is
    /// strongly recommended above a few MHz of SCK: errata for these devices describe RAM
    /// corruption at high SCK with simultaneous bus activity. The cost is 3 extra bytes per
    /// SFR access, 4 per RAM access, and the CRC computation.
    pub fn set_crc_protection(&mut self, enabled: bool) {
        self.crc_protection = enabled;
    }

    /// Whether CRC-protected SPI transfers are enabled.
    pub fn crc_protection(&self) -> bool {
        self.crc_protection
    }

    /// Releases ownership of the SPI resources
    pub fn free(self) -> SPI {
        self.spi
    }

    /// Performs a software reset of the MCP2518FD chip over SPI, leaving it in Configuration
    /// mode with the oscillator running.
    ///
    /// Neither the mode request nor the RESET instruction work while the device is in Sleep
    /// mode, so the oscillator is first re-enabled by clearing OSC.OSCDIS (this also wakes a
    /// device in Low Power Mode, and resets PLLEN/SCLKDIV to their defaults, which
    /// [`MCP2518FD::configure_osc`] restores). The RESET instruction is only defined for a
    /// device in Configuration mode (datasheet section 4.1.1), so that mode is requested next.
    /// A failure to get there is ignored because a reset is the recovery path for a wedged
    /// controller; the state is verified after the reset instead. Times out after roughly
    /// 30 ms if the oscillator does not report ready or the device is not in Configuration
    /// mode.
    pub async fn reset(&mut self, delay: &mut impl DelayNs) -> Result<(), ConfigError> {
        // Power-on defaults: clock enabled, PLL off, SYSCLK undivided, CLKO divided by 10.
        let mut osc = OscillatorControlRegister(0);
        osc.set_oscdis(false);
        osc.set_clkodiv(0b11);
        self.write_register(osc).await?;

        let _ = self.set_op_mode(OperationMode::Configuration, delay).await;

        let instruction = Instruction(OpCode::RESET);

        self.spi
            .write(&instruction.0.to_be_bytes())
            .await
            .map_err(|_| Error::SPIWrite)?;

        const POLL_INTERVAL_US: u32 = 500;
        const MAX_ATTEMPTS: usize = 60;

        for attempt in 0..MAX_ATTEMPTS {
            let osc = self.read_register::<OscillatorControlRegister>().await?;

            if osc.oscrdy() {
                let c1con = self.read_register::<CanControlRegister>().await?;

                if c1con.opmode() == OperationMode::Configuration {
                    return Ok(());
                }
            }

            if attempt < MAX_ATTEMPTS - 1 {
                delay.delay_us(POLL_INTERVAL_US).await;
            }
        }

        Err(ConfigError::ConfigurationModeTimeout)
    }

    /// Does a full configuration sequence of the chip using the provided
    /// settings. This function puts the chip into configuration mode if it
    /// isn't already, verifies that SPI communication with the chip is
    /// working, and writes to all the necessary configuration registers.
    ///
    /// You may want to reset the chip before calling this method. See
    /// [`MCP2518FD::reset`] for more information.
    ///
    /// The data_bits_to_match field must be within 1..=18 if it is `Some`. A value of Some(0) will be interpreted the same as None, and
    pub async fn configure(
        &mut self,
        settings: settings::Settings,
        delay: &mut impl DelayNs,
    ) -> Result<(), ConfigError> {
        self.set_op_mode(OperationMode::Configuration, delay)
            .await
            .map_err(|_| ConfigError::ConfigurationModeTimeout)?;

        self.verify_spi_communications().await?;

        self.configure_osc(settings.oscillator, delay).await?;
        self.configure_io(settings.io_configuration).await?;
        self.configure_bit_timing(settings.bit_time_configuration)
            .await?;
        self.configure_tx_event_fifo(settings.tx_event_fifo).await?;
        self.configure_tx_queue(settings.tx_queue).await?;

        if let Some(time_base_counter) = settings.time_base_counter {
            self.modify_register(|mut tscon: TimeStampControlRegister| {
                tscon.set_tbcpre(
                    time_base_counter
                        .prescaler
                        .min(settings::TimeBaseCounterConfiguration::MAX_PRESCALER),
                );
                tscon.set_tbcen(true);
                tscon
            })
            .await?;
        }

        if let Some(dncnt) = settings.data_bits_to_match {
            self.modify_register(|mut cicon: CanControlRegister| {
                cicon.set_dncnt(dncnt);
                cicon
            })
            .await?;
        }

        self.modify_register(|mut cicon: CanControlRegister| {
            cicon.set_rtxat(true);
            cicon
        })
        .await?;

        self.modify_register(|mut ciint: InterruptRegister| {
            ciint.set_rxie(true);
            ciint.set_txie(true);

            if settings.enable_can_error_interrupts {
                ciint.set_ivmie(true);
                ciint.set_cerrie(true);
                ciint.set_serrie(true);
            }

            if settings.enable_spi_error_interrupt {
                ciint.set_spicrcie(true);
            }

            if settings.enable_ecc_error_interrupt {
                ciint.set_eccie(true);
            }

            if settings
                .time_base_counter
                .is_some_and(|tbc| tbc.overflow_interrupt)
            {
                ciint.set_tbcie(true);
            }

            ciint
        })
        .await?;

        Ok(())
    }

    pub async fn get_op_mode(&mut self) -> Result<OperationMode, Error> {
        let c1con = self.read_register::<CanControlRegister>().await?;
        Ok(c1con.opmode())
    }

    /// Changes the operating mode of the chip.
    ///
    /// Leaving Normal mode waits for the bus to become idle, which can take a full frame
    /// (about 75 ms for a CAN FD frame at 10 kbit/s), so this polls for up to 100 ms.
    pub async fn set_op_mode(
        &mut self,
        op_mode: OperationMode,
        delay: &mut impl DelayNs,
    ) -> Result<(), ConfigError> {
        self.modify_register(|mut c1con: CanControlRegister| {
            c1con.set_opmode(op_mode);
            c1con
        })
        .await?;

        const POLL_INTERVAL_US: u32 = 200;
        const MAX_ATTEMPTS: usize = 500;

        for attempt in 0..MAX_ATTEMPTS {
            let c1con = self.read_register::<CanControlRegister>().await?;

            if c1con.opmode() == op_mode {
                return Ok(());
            }

            if attempt < MAX_ATTEMPTS - 1 {
                delay.delay_us(POLL_INTERVAL_US).await;
            }
        }

        Err(ConfigError::ChangeOpModeTimeout)
    }

    pub async fn configure_osc(
        &mut self,
        oscillator_settings: OscillatorConfiguration,
        delay: &mut impl DelayNs,
    ) -> Result<(), ConfigError> {
        self.modify_register(|mut osc: OscillatorControlRegister| {
            // If enabled, system clock comes from 10x PLL, otherwise it comes directly from the XTAL
            osc.set_pllen(match oscillator_settings.pll {
                Pll::On => true,
                Pll::Off => false,
            });

            // Whether or not to divide the system clock by 2
            osc.set_sclkdiv(match oscillator_settings.divider {
                SysClkDivider::DivByOne => false,
                SysClkDivider::DivByTwo => true,
            });

            // Enable the clock
            osc.set_oscdis(false);

            osc
        })
        .await?;

        if let settings::Pll::On = oscillator_settings.pll {
            const POLL_INTERVAL_US: u32 = 500;
            const MAX_ATTEMPTS: usize = 60;

            // Wait up to 30 ms for the PLL to lock
            for attempt in 0..MAX_ATTEMPTS {
                let osc = self.read_register::<OscillatorControlRegister>().await?;

                if osc.pllrdy() {
                    return Ok(());
                }

                if attempt < MAX_ATTEMPTS - 1 {
                    delay.delay_us(POLL_INTERVAL_US).await;
                }
            }

            return Err(ConfigError::PLLNotReady);
        }

        Ok(())
    }

    pub async fn configure_io(&mut self, io_config: IoConfiguration) -> Result<(), ConfigError> {
        let mut iocon = self.read_register::<IoControlRegister>().await?;

        iocon.set_xstbyen(io_config.enable_tx_standby_pin);
        iocon.set_txcanod(io_config.tx_can_open_drain);
        iocon.set_sof(io_config.start_of_frame_on_clko);
        iocon.set_intod(io_config.interrupt_pin_open_drain);

        self.write_io_control(iocon).await?;

        Ok(())
    }

    /// Writes IOCON one byte at a time, skipping the GPIO pin status byte.
    ///
    /// The datasheet requires single-byte writes for IOCON (Table 3-1, Note 2). Per errata
    /// DS80000789E item 5, a write that covers byte 2 (bits 23:16, the read-only GPIO pin
    /// status) clears LAT0/LAT1, so that byte is never written.
    pub async fn write_io_control(&mut self, iocon: IoControlRegister) -> Result<(), Error> {
        let bytes = u32::from(iocon).to_le_bytes();

        for byte_offset in [0u8, 1, 3] {
            self.write_sfr_byte(&SFRAddress::IOCON, byte_offset, bytes[byte_offset as usize])
                .await?;
        }

        Ok(())
    }

    pub async fn configure_bit_timing(
        &mut self,
        bit_time_config: BitTimeConfiguration,
    ) -> Result<(), ConfigError> {
        self.modify_register(|mut cinbtcfg: NominalBitTimeConfigurationRegister| {
            cinbtcfg.set_brp(bit_time_config.nominal.baud_rate_prescaler);
            cinbtcfg.set_tseg1(bit_time_config.nominal.time_segment_1);
            cinbtcfg.set_tseg2(bit_time_config.nominal.time_segment_2.value());
            cinbtcfg.set_sjw(bit_time_config.nominal.synchronization_jump_width.value());

            cinbtcfg
        })
        .await?;

        self.modify_register(|mut cidbtcfg: DataBitTimeConfigurationRegister| {
            cidbtcfg.set_brp(bit_time_config.data.baud_rate_prescaler);
            cidbtcfg.set_tseg1(bit_time_config.data.time_segment_1.value());
            cidbtcfg.set_tseg2(bit_time_config.data.time_segment_2.value());
            cidbtcfg.set_sjw(bit_time_config.data.synchronization_jump_width.value());

            cidbtcfg
        })
        .await?;

        self.modify_register(|mut citdc: TransmitterDelayCompensationRegister| {
            citdc.set_tdcmod(TransmitterDelayCompensationMode::Automatic);
            citdc.set_tdco(
                bit_time_config
                    .data
                    .transmitter_delay_compensation_offset
                    .value(),
            );
            citdc.set_tdcv(0);

            citdc
        })
        .await?;

        Ok(())
    }

    /// Enables/Disables the transmit event FIFO by setting C1CON.STEF and appropriate C1TEFCON bits.
    /// Be aware that fifo_size MUST be <= 32 and > 0, any other values will be clamped to 32.
    ///
    /// Also please keep in mind that the total RAM size is 2K and this code does absolutely
    /// zero validation that your configuration is under this limit. The documentation recommends
    /// configuring the TEF first, then TEQ, then FIFOs as necessary.
    pub async fn configure_tx_event_fifo(
        &mut self,
        tx_event_fifo_config: Option<TxEventFifoConfiguration>,
    ) -> Result<(), ConfigError> {
        self.modify_register(|mut c1con: CanControlRegister| {
            c1con.set_stef(tx_event_fifo_config.is_some());
            c1con
        })
        .await?;

        if let Some(config) = tx_event_fifo_config {
            self.modify_register(|mut tef_control: TxEventFifoControlRegister| {
                tef_control.set_fifo_size(config.fifo_size);

                tef_control.set_teftsen(config.enable_timestamps);

                tef_control.set_tefovie(config.enable_fifo_overflow_interrupt);
                tef_control.set_teffie(config.enable_fifo_full_interrupt);
                tef_control.set_tefhie(config.enable_fifo_half_full_interrupt);
                tef_control.set_tefneie(config.enable_fifo_not_empty_interrupt);

                tef_control
            })
            .await?;
        }

        Ok(())
    }

    /// Enables/Disables the transmit queue by setting C1CON.TXEN and appropriate C1TXQCON bits.
    /// Be aware that fifo_size MUST be <= 32 and > 0, any other values will be clamped to 32.
    ///
    /// Also please keep in mind that the total RAM size is 2K and this code does absolutely
    /// zero validation that your configuration is under this limit. The documentation recommends
    /// configuring the TEF first, then TEQ, then FIFOs as necessary.
    pub async fn configure_tx_queue(
        &mut self,
        tx_queue_config: Option<TxQueueConfiguration>,
    ) -> Result<(), ConfigError> {
        self.modify_register(|mut c1con: CanControlRegister| {
            c1con.set_txqen(tx_queue_config.is_some());
            c1con
        })
        .await?;

        if let Some(config) = tx_queue_config {
            self.modify_register(|mut tx_queue_control: TxQueueControlRegister| {
                tx_queue_control.set_retransmission_attempts(config.retransmission_attempts);
                tx_queue_control.set_txpri(config.message_priority);
                tx_queue_control.set_fifo_size(config.fifo_size);
                tx_queue_control.set_payload_size(config.payload_size);

                tx_queue_control.set_txatie(config.enable_transmit_attempts_exhausted_interrupt);
                tx_queue_control.set_txqeie(config.enable_queue_empty_interrupt);
                tx_queue_control.set_txqnie(config.enable_queue_not_full_interrupt);

                tx_queue_control
            })
            .await?;
        }

        Ok(())
    }

    /// Configures a FIFO based on the settings provided. As per documentation, a single FIFO must
    /// be dedicated to RX or TX and all objects in that queue must have the same payload size.
    pub async fn configure_fifo(
        &mut self,
        fifo_number: FifoNumber,
        fifo_config: FifoConfiguration,
    ) -> Result<(), Error> {
        self.modify_repeated_register(fifo_number, |mut fifo_control: FifoControlRegister| {
            fifo_control.set_fifo_size(fifo_config.fifo_size);
            fifo_control.set_payload_size(fifo_config.payload_size);

            match fifo_config.mode {
                settings::FifoMode::Transmit(TxFifoConfiguration {
                    priority,
                    retransmission_attempts,
                    enable_auto_rtr,
                    enable_transmit_attempts_exhausted_interrupt,
                    enable_fifo_empty_interrupt,
                    enable_fifo_half_empty_interrupt,
                    enable_fifo_not_full_interrupt,
                }) => {
                    fifo_control.set_txen(true);

                    fifo_control.set_txpri(priority);
                    fifo_control.set_retransmission_attempts(retransmission_attempts);
                    fifo_control.set_rtren(enable_auto_rtr);
                    fifo_control.set_txatie(enable_transmit_attempts_exhausted_interrupt);
                    fifo_control.set_tferffie(enable_fifo_empty_interrupt);
                    fifo_control.set_tfhrfhie(enable_fifo_half_empty_interrupt);
                    fifo_control.set_tfnrfnie(enable_fifo_not_full_interrupt);
                }
                settings::FifoMode::Receive(RxFifoConfiguration {
                    enable_message_timestamps,
                    enable_fifo_overflow_interrupt,
                    enable_fifo_full_interrupt,
                    enable_fifo_half_full_interrupt,
                    enable_fifo_not_empty_interrupt,
                }) => {
                    fifo_control.set_txen(false);

                    fifo_control.set_rxtsen(enable_message_timestamps);
                    fifo_control.set_rxovie(enable_fifo_overflow_interrupt);
                    fifo_control.set_tferffie(enable_fifo_full_interrupt);
                    fifo_control.set_tfhrfhie(enable_fifo_half_full_interrupt);
                    fifo_control.set_tfnrfnie(enable_fifo_not_empty_interrupt);
                }
            }

            fifo_control
        })
        .await?;

        Ok(())
    }

    /// Configures one of the 32 acceptance filters. If the filter_config is
    /// None, the filter will be disabled instead.
    ///
    /// Filters can be configured to accept only standard frames, only
    /// extended frames, or both standard and extended frames. If either the
    /// filter_bits or mask_bits fields are set as MessageId::Standard, the
    /// corresponding EID bits will be set to 0.
    ///
    /// When receiving standard frames, the EID component of the filter can be
    /// used to match against (up to) the first 18 bits of the message's data
    /// segment. The number of bits used is configured by `CiCON.DNCNT`. See
    /// the family reference manual for a more detailed description of this
    /// mechanism.
    pub async fn configure_filter(
        &mut self,
        filter_number: FilterNumber,
        filter_config: Option<FilterConfiguration>,
    ) -> Result<(), Error> {
        let (control_register_number, filter_index) = filter_number.get_control_register();

        // We need to disable the filter no matter what to configure it
        self.modify_repeated_register(
            control_register_number,
            |mut control: FilterControlRegister| {
                control.set_enabled(filter_index, false);
                control
            },
        )
        .await?;

        // If we are just disabling it, then we are done here
        let Some(filter_config) = filter_config else {
            return Ok(());
        };

        // Set filter object bits and filter mode
        self.modify_repeated_register(
            filter_number,
            |mut object_register: FilterObjectRegister| {
                match filter_config.filter_bits {
                    Id::Standard(id) => {
                        object_register.set_sid(id.as_raw());
                        object_register.set_eid(0);
                    }
                    Id::Extended(id) => {
                        object_register.set_sid(id.standard_id().as_raw());
                        object_register
                            .set_eid(id.as_raw() & ((!(StandardId::MAX.as_raw() as u32)) >> 11));
                    }
                }

                // 0b000S_SSSS_SSSS_SSEE_EEEE_EEEE_EEEE_EEEE
                //                     0b0000_0111_1111_1111
                //                     0b1111_1000_0000_0000
                // 0b0000_0000_0000_0000_1111_1000_0000_0000
                // 0b0000_0000_0000_0000_0000_0000_0001_1111

                object_register.set_exide(match filter_config.mode {
                    FilterMatchMode::StandardOnly | FilterMatchMode::Both => false,
                    FilterMatchMode::ExtendedOnly => true,
                });

                object_register
            },
        )
        .await?;

        // Set the mask bits and exclusion mode
        self.modify_repeated_register(filter_number, |mut mask_register: MaskRegister| {
            match filter_config.mask_bits {
                Id::Standard(id) => {
                    mask_register.set_msid(id.as_raw());
                    mask_register.set_meid(0);
                }
                Id::Extended(id) => {
                    mask_register.set_msid(id.standard_id().as_raw());
                    mask_register
                        .set_meid(id.as_raw() & ((!(StandardId::MAX.as_raw() as u32)) >> 11));
                }
            }

            mask_register.set_mide(match filter_config.mode {
                FilterMatchMode::Both => false,
                FilterMatchMode::StandardOnly | FilterMatchMode::ExtendedOnly => true,
            });

            mask_register
        })
        .await?;

        // Set the BP and reenable the filter
        self.modify_repeated_register(
            control_register_number,
            |mut control: FilterControlRegister| {
                control.set_buffer_pointer(filter_index, filter_config.buffer_pointer);
                control.set_enabled(filter_index, true);
                control
            },
        )
        .await?;

        Ok(())
    }

    /* Transmit and Receive Functions */

    /// Writes byte 1 of a FIFO/TXQ/TEF control register, which holds only UINC (bit 8), TXREQ
    /// (bit 9) and FRESET (bit 10).
    ///
    /// A single-byte write avoids re-writing the configuration bits in the other bytes and
    /// halves the SPI traffic of a read-modify-write. Callers must pass the current TXREQ state
    /// (or `true` to request transmission): writing TXREQ as 0 while it is set requests an
    /// abort of the pending transmission (Registers 3-26 and 3-29).
    async fn write_fifo_control_byte(
        &mut self,
        address: &SFRAddress,
        increment: bool,
        request_transmission: bool,
    ) -> Result<(), Error> {
        const UINC: u8 = 1 << 0;
        const TXREQ: u8 = 1 << 1;

        let mut value = 0;
        if increment {
            value |= UINC;
        }
        if request_transmission {
            value |= TXREQ;
        }

        self.write_sfr_byte(address, 1, value).await
    }

    /// Pushes a new message into the TXQ without setting the TXREQ bit to
    /// request transmission.
    ///
    /// Use this function only if you need to queue multiple messages before
    /// transmitting all at once. To push a single message and immediately
    /// request transmission, use [`MCP2518FD::tx_queue_transmit_message`].
    pub async fn tx_queue_push_message(&mut self, message: &TxMessage) -> Result<(), Error> {
        self.tx_queue_push_message_inner(message, false).await
    }

    async fn tx_queue_push_message_inner(
        &mut self,
        message: &TxMessage,
        request_transmission: bool,
    ) -> Result<(), Error> {
        /* Make sure TXQ is enabled */

        if !self.read_register::<CanControlRegister>().await?.txqen() {
            return Err(Error::TxQueueDisabled);
        }

        let control_register = self.read_register::<TxQueueControlRegister>().await?;

        /* Make sure FIFO is big enough */

        if control_register.payload_size().num_bytes() < message.data().len() {
            return Err(Error::FifoTooSmall);
        }

        /* Make sure FIFO is not full */

        let status_register = self.read_register::<TxQueueStatusRegister>().await?;

        if !status_register.txqnif() {
            return Err(Error::FifoFull);
        }

        /* Write message to RAM */

        let ram_address = self
            .read_repeated_register::<UserAddressRegister>(UserAddressKind::TxQueue)
            .await?
            .calculate_ram_address();

        let (length, bytes) = message.as_bytes();

        // The reading length has to be a multiple of 4 thus we round up the data_len
        let data = &bytes[..round_up_spi_transfer_size(length)];

        self.write_ram(ram_address as u16, data).await?;

        /* Increment the head, keeping any pending TXREQ */

        self.write_fifo_control_byte(
            &SFRAddress::C1TXQCON,
            true,
            request_transmission || control_register.txreq(),
        )
        .await
    }

    /// Pushes to the TXQ without reading any status registers or performing
    /// size checks. Callers of this function should be careful to check that
    /// there is room in the TXQ. Otherwise, the oldest message will be
    /// overriden.
    pub async fn tx_queue_push_message_unchecked(
        &mut self,
        message: &TxMessage,
    ) -> Result<(), Error> {
        /* Write message to RAM */

        let ram_address = self
            .read_repeated_register::<UserAddressRegister>(UserAddressKind::TxQueue)
            .await?
            .calculate_ram_address();

        let (length, bytes) = message.as_bytes();

        // The reading length has to be a multiple of 4 thus we round up the data_len
        let data = &bytes[..round_up_spi_transfer_size(length)];

        self.write_ram(ram_address as u16, data).await?;

        /* Increment the head, keeping any pending TXREQ */

        let control_register = self.read_register::<TxQueueControlRegister>().await?;

        self.write_fifo_control_byte(&SFRAddress::C1TXQCON, true, control_register.txreq())
            .await
    }

    /// Requests transmission of all messages in the TXQ by setting the TXREQ
    /// bit.
    ///
    /// Use this function only if you already previously queued one or more
    /// messages with [`MCP2518FD::tx_queue_push_message`]. To push a single
    /// message and immediately request transmission, prefer
    /// [`MCP2518FD::tx_queue_transmit_message`].
    pub async fn tx_queue_request_transmission(&mut self) -> Result<(), Error> {
        self.write_fifo_control_byte(&SFRAddress::C1TXQCON, false, true)
            .await
    }

    /// Pushes a message into the TXQ and immediately requests transmission by
    /// setting the TXREQ bit.
    ///
    /// To push multiple messages before requesting transmission, see
    /// [`MCP2518FD::tx_queue_push_message`] and
    /// [`MCP2518FD::tx_queue_request_transmission`].
    pub async fn tx_queue_transmit_message(&mut self, message: &TxMessage) -> Result<(), Error> {
        self.tx_queue_push_message_inner(message, true).await
    }

    /// Fetches the status of the TXQ to determine whether it is empty
    pub async fn tx_queue_is_empty(&mut self) -> Result<bool, Error> {
        let status = self.read_register::<TxQueueStatusRegister>().await?;

        Ok(status.txqeif())
    }

    /// Fetches the status of the TXQ to determine whether it is full
    pub async fn tx_queue_is_full(&mut self) -> Result<bool, Error> {
        let status = self.read_register::<TxQueueStatusRegister>().await?;

        Ok(!status.txqnif())
    }

    /// Pushes a new message into the given TX FIFO without setting the TXREQ
    /// bit to request transmission.
    ///
    /// Use this function only if you need to queue multiple messages before
    /// transmitting all at once. To push a single message and immediately
    /// request transmission, use [`MCP2518FD::tx_fifo_transmit_message`].
    pub async fn tx_fifo_push_message(
        &mut self,
        fifo_number: FifoNumber,
        message: &TxMessage,
    ) -> Result<(), Error> {
        self.tx_fifo_push_message_inner(fifo_number, message, false)
            .await
    }

    async fn tx_fifo_push_message_inner(
        &mut self,
        fifo_number: FifoNumber,
        message: &TxMessage,
        request_transmission: bool,
    ) -> Result<(), Error> {
        let control_register = self
            .read_repeated_register::<FifoControlRegister>(fifo_number)
            .await?;

        /* Make sure it's a transmit FIFO */

        if !control_register.txen() {
            return Err(Error::FifoNotTx);
        }

        /* Make sure FIFO is big enough */

        if control_register.payload_size().num_bytes() < message.data().len() {
            return Err(Error::FifoTooSmall);
        }

        /* Make sure FIFO is not full */

        let status_register = self
            .read_repeated_register::<FifoStatusRegister>(fifo_number)
            .await?;

        if !status_register.tfnrfnif() {
            return Err(Error::FifoFull);
        }

        /* Write message to RAM */

        let ram_address = self
            .read_repeated_register::<UserAddressRegister>(UserAddressKind::Fifo(fifo_number))
            .await?
            .calculate_ram_address();

        let (length, bytes) = message.as_bytes();

        // The reading length has to be a multiple of 4 thus we round up the data_len
        let data = &bytes[..round_up_spi_transfer_size(length)];

        self.write_ram(ram_address as u16, data).await?;

        /* Increment the head, keeping any pending TXREQ */

        self.write_fifo_control_byte(
            &FifoControlRegister::get_address_for(fifo_number),
            true,
            request_transmission || control_register.txreq(),
        )
        .await
    }

    /// Requests transmission of all messages in the given TX FIFO by setting
    /// the TXREQ bit.
    ///
    /// Use this function only if you already previously queued one or more
    /// messages with [`MCP2518FD::tx_fifo_push_message`]. To push a single
    /// message and immediately request transmission, prefer
    /// [`MCP2518FD::tx_fifo_transmit_message`].
    pub async fn tx_fifo_request_transmission(
        &mut self,
        fifo_number: FifoNumber,
    ) -> Result<(), Error> {
        self.write_fifo_control_byte(
            &FifoControlRegister::get_address_for(fifo_number),
            false,
            true,
        )
        .await
    }

    /// Pushes a message into the given TX FIFO and immediately requests
    /// transmission by setting the TXREQ bit.
    ///
    /// To push multiple messages before requesting transmission, see
    /// [`MCP2518FD::tx_fifo_push_message`] and
    /// [`MCP2518FD::tx_fifo_request_transmission`].
    pub async fn tx_fifo_transmit_message(
        &mut self,
        fifo_number: FifoNumber,
        message: &TxMessage,
    ) -> Result<(), Error> {
        self.tx_fifo_push_message_inner(fifo_number, message, true)
            .await
    }

    /// Fetches the status of the TX FIFO to determine whether it is empty
    pub async fn tx_fifo_is_empty(&mut self, fifo_number: FifoNumber) -> Result<bool, Error> {
        let status = self
            .read_repeated_register::<FifoStatusRegister>(fifo_number)
            .await?;

        Ok(status.tferffif())
    }

    /// Fetches the status of the TX FIFO to determine whether it is full
    pub async fn tx_fifo_is_full(&mut self, fifo_number: FifoNumber) -> Result<bool, Error> {
        let status = self
            .read_repeated_register::<FifoStatusRegister>(fifo_number)
            .await?;

        Ok(!status.tfnrfnif())
    }

    /// Gets the value of TXAT for this FIFO and clears it
    pub async fn tx_fifo_transmission_attempts_exhausted(
        &mut self,
        fifo_number: FifoNumber,
    ) -> Result<bool, Error> {
        let status = self
            .clear_repeated_register_flags(fifo_number, |status: &mut FifoStatusRegister| {
                status.clear_txatif()
            })
            .await?;

        Ok(status.txatif())
    }

    /// Checks to see if there are any messages in the TEF
    pub async fn tx_event_fifo_has_next(&mut self) -> Result<bool, Error> {
        let status_register = self.read_register::<TxEventFifoStatusRegister>().await?;

        Ok(status_register.tefneif())
    }

    /// If there is a message available in the TEF it will be read but the FIFO
    /// tail pointer will **NOT** be incremented
    ///
    /// Unless you have a specific use case for this, you most likely want to
    /// use [`MCP2518FD::tx_event_fifo_get_next`]
    pub async fn tx_event_fifo_peek_next(&mut self) -> Result<Option<TxEventObject>, Error> {
        /* Make sure there is data to read */

        if !self.tx_event_fifo_has_next().await? {
            return Ok(None);
        }

        /* Get the address of the next object */

        let ram_address = self
            .read_repeated_register::<UserAddressRegister>(UserAddressKind::TxEventFifo)
            .await?
            .calculate_ram_address();

        /* Read the header and, if enabled, the timestamp in one transfer */

        let control_register = self.read_register::<TxEventFifoControlRegister>().await?;
        let with_timestamp = control_register.teftsen();

        let mut buf = [0u8; 12];
        let read_len = if with_timestamp { 12 } else { 8 };

        self.read_ram(ram_address as u16, &mut buf[..read_len])
            .await?;

        Ok(Some(TxEventObject {
            header: TxHeader([
                u32::from_le_bytes(buf[0..4].try_into().unwrap()),
                u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            ]),
            timestamp: with_timestamp.then(|| u32::from_le_bytes(buf[8..12].try_into().unwrap())),
        }))
    }

    /// If there is a message available in the TEF it will be read and the FIFO
    /// tail pointer will be incremented to allow for the next read operation
    ///
    /// To only check if a message is available without pulling it from the
    /// FIFO, see [`MCP2518FD::tx_event_fifo_has_next`] and
    /// [`MCP2518FD::tx_event_fifo_peek_next`]
    pub async fn tx_event_fifo_get_next(&mut self) -> Result<Option<TxEventObject>, Error> {
        let obj = self.tx_event_fifo_peek_next().await?;

        let Some(obj) = obj else {
            return Ok(None);
        };

        self.write_fifo_control_byte(&SFRAddress::C1TEFCON, true, false)
            .await?;

        Ok(Some(obj))
    }

    /// Checks to see if there are any messages in the given receive FIFO
    pub async fn rx_fifo_has_next(&mut self, fifo_number: FifoNumber) -> Result<bool, Error> {
        /* Make sure it's a receive FIFO */

        let control_register = self
            .read_repeated_register::<FifoControlRegister>(fifo_number)
            .await?;

        if control_register.txen() {
            return Err(Error::FifoNotRx);
        }

        /* Check is the FIFO has any messages in it */

        let status_register = self
            .read_repeated_register::<FifoStatusRegister>(fifo_number)
            .await?;

        Ok(status_register.tfnrfnif())
    }

    /// If there is a message available in the given RX FIFO it will be read,
    /// but the FIFO head pointer will **NOT** be incremented
    ///
    /// Unless you have a specific use case for this, you most likely want to
    /// use [`MCP2518FD::rx_fifo_get_next`]
    pub async fn rx_fifo_peek_next(
        &mut self,
        fifo_number: FifoNumber,
    ) -> Result<Option<RxMessage>, Error> {
        /* Make sure it's a receive FIFO with data to read */

        let control_register = self
            .read_repeated_register::<FifoControlRegister>(fifo_number)
            .await?;

        if control_register.txen() {
            return Err(Error::FifoNotRx);
        }

        let status_register = self
            .read_repeated_register::<FifoStatusRegister>(fifo_number)
            .await?;

        if !status_register.tfnrfnif() {
            return Ok(None);
        }

        /* Get the address of the next object */

        let ram_address = self
            .read_repeated_register::<UserAddressRegister>(UserAddressKind::Fifo(fifo_number))
            .await?
            .calculate_ram_address();

        /* Read the whole message object (header, optional timestamp, payload) in one transfer.
         * The payload area is PLSIZE bytes regardless of the DLC. */

        const HEADER_LEN: usize = 8;
        const TIMESTAMP_LEN: usize = 4;

        let with_timestamp = control_register.rxtsen();
        let payload_len = control_register.payload_size().num_bytes();
        let data_offset = HEADER_LEN + if with_timestamp { TIMESTAMP_LEN } else { 0 };

        let mut buf = [0u8; HEADER_LEN + TIMESTAMP_LEN + MAX_FD_BUFFER_SIZE];

        self.read_ram(ram_address as u16, &mut buf[..data_offset + payload_len])
            .await?;

        let header = RxHeader([
            u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            u32::from_le_bytes(buf[4..8].try_into().unwrap()),
        ]);

        let timestamp = with_timestamp.then(|| u32::from_le_bytes(buf[8..12].try_into().unwrap()));

        // A DLC larger than the FIFO's payload size (DLCMM) leaves the excess bytes unavailable.
        let data_len = len_for_dlc(header.dlc(), header.fdf())
            .unwrap_or(0)
            .min(payload_len);

        Ok(Some(
            RxMessage::new(header, timestamp, &buf[data_offset..data_offset + data_len]).unwrap(),
        ))
    }

    /// If there is a message available in the given RX FIFO it will be read,
    /// and the FIFO head pointer will be incremented to allow for the next
    /// read operation
    ///
    /// To only check if a message is available without pulling it from the
    /// FIFO, see [`MCP2518FD::rx_fifo_has_next`] and
    /// [`MCP2518FD::rx_fifo_peek_next`]
    pub async fn rx_fifo_get_next(
        &mut self,
        fifo_number: FifoNumber,
    ) -> Result<Option<RxMessage>, Error> {
        let msg = self.rx_fifo_peek_next(fifo_number).await?;

        let Some(msg) = msg else {
            return Ok(None);
        };

        // TXREQ has no effect on a receive FIFO, so it is safe to write as 0 here.
        self.write_fifo_control_byte(
            &FifoControlRegister::get_address_for(fifo_number),
            true,
            false,
        )
        .await?;

        Ok(Some(msg))
    }

    /* Interrupt related operations */

    pub async fn get_highest_interrupt_codes(&mut self) -> Result<InterruptCodeRegister, Error> {
        self.read_register::<InterruptCodeRegister>().await
    }

    pub async fn get_top_level_interrupt_statuses(&mut self) -> Result<InterruptRegister, Error> {
        self.read_register_crc::<InterruptRegister>().await
    }

    pub async fn get_rx_interrupt_statuses(&mut self) -> Result<RxInterruptStatusRegister, Error> {
        self.read_register::<RxInterruptStatusRegister>().await
    }

    pub async fn get_rx_overflow_interrupt_statuses(
        &mut self,
    ) -> Result<RxOverflowInterruptStatusRegister, Error> {
        self.read_register::<RxOverflowInterruptStatusRegister>()
            .await
    }

    pub async fn get_tx_interrupt_statuses(&mut self) -> Result<TxInterruptStatusRegister, Error> {
        self.read_register::<TxInterruptStatusRegister>().await
    }

    pub async fn get_tx_attempt_interrupt_statuses(
        &mut self,
    ) -> Result<TxAttemptInterruptStatusRegister, Error> {
        self.read_register::<TxAttemptInterruptStatusRegister>()
            .await
    }

    pub async fn get_time_base_counter(&mut self) -> Result<u32, Error> {
        // NOTE: Errata 1 in DS80000984A recomments to use READ_CRC for this
        // register
        Ok(self
            .read_register_crc::<TimeBaseCounterRegister>()
            .await?
            .tbc())
    }

    /* Generic register ops with mapping */

    /// Clears hardware-set flags in a repeated register without the read-modify-write race.
    ///
    /// See [`MCP2518FD::clear_register_flags`].
    pub async fn clear_repeated_register_flags<R, F>(
        &mut self,
        index: R::Index,
        clear: F,
    ) -> Result<R, Error>
    where
        R: RepeatedRegister + ClearableFlags + From<u32> + Into<u32>,
        F: FnOnce(&mut R),
    {
        let address = R::get_address_for(index);
        let raw = self.read_sfr(&address).await?;

        let mut register = R::from(raw | R::CLEARABLE_FLAGS);
        clear(&mut register);
        self.write_sfr(&address, register.into()).await?;

        Ok(R::from(raw))
    }

    /// Clears hardware-set flags in a register without the read-modify-write race.
    ///
    /// `HS/C` flags are cleared by writing 0 and unaffected by writing 1. The register is read,
    /// every clearable flag is set to 1 in the copy, `clear` resets the ones to clear, and the
    /// result is written back. Flags set by hardware between the read and the write therefore
    /// survive, unlike with [`MCP2518FD::modify_register`]. Returns the register as it was read,
    /// so the caller can inspect which flags were pending.
    pub async fn clear_register_flags<R, F>(&mut self, clear: F) -> Result<R, Error>
    where
        R: Register + ClearableFlags + From<u32> + Into<u32>,
        F: FnOnce(&mut R),
    {
        let address = R::get_address();
        let raw = self.read_sfr(&address).await?;

        let mut register = R::from(raw | R::CLEARABLE_FLAGS);
        clear(&mut register);
        self.write_sfr(&address, register.into()).await?;

        Ok(R::from(raw))
    }

    /// Reads a repeated register, applies `transform` and writes the result back.
    ///
    /// Do not use this to clear hardware-set flags: any flag set between the read and the write
    /// is written back as 0 and lost. Use [`MCP2518FD::clear_repeated_register_flags`] instead.
    pub async fn modify_repeated_register<R, F>(
        &mut self,
        index: R::Index,
        transform: F,
    ) -> Result<(), Error>
    where
        R: RepeatedRegister + From<u32> + Into<u32>,
        F: FnOnce(R) -> R,
    {
        let register = self.read_repeated_register::<R>(index).await?;

        self.write_repeated_register::<R>(index, transform(register))
            .await
    }

    pub async fn read_repeated_register<R>(&mut self, index: R::Index) -> Result<R, Error>
    where
        R: RepeatedRegister + From<u32>,
    {
        let address = R::get_address_for(index);

        self.read_sfr(&address).await.map(R::from)
    }

    pub async fn write_repeated_register<R>(
        &mut self,
        index: R::Index,
        value: R,
    ) -> Result<(), Error>
    where
        R: RepeatedRegister + Into<u32>,
    {
        let address = R::get_address_for(index);

        self.write_sfr(&address, value.into()).await
    }

    /// Reads a register, applies `transform` and writes the result back.
    ///
    /// Do not use this to clear hardware-set flags: any flag set between the read and the write
    /// is written back as 0 and lost. Use [`MCP2518FD::clear_register_flags`] instead.
    pub async fn modify_register<R, F>(&mut self, transform: F) -> Result<(), Error>
    where
        R: Register + From<u32> + Into<u32>,
        F: FnOnce(R) -> R,
    {
        let register = self.read_register::<R>().await?;

        self.write_register::<R>(transform(register)).await
    }

    pub async fn read_register<R>(&mut self) -> Result<R, Error>
    where
        R: Register + From<u32>,
    {
        let address = R::get_address();

        self.read_sfr(&address).await.map(R::from)
    }

    pub async fn write_register<R>(&mut self, value: R) -> Result<(), Error>
    where
        R: Register + Into<u32>,
    {
        let address = R::get_address();

        self.write_sfr(&address, value.into()).await
    }

    /// Reads a register with `READ_CRC` regardless of the CRC protection setting, retrying on
    /// a CRC mismatch.
    pub async fn read_register_crc<R>(&mut self) -> Result<R, Error>
    where
        R: Register + From<u32>,
    {
        let address = R::get_address();

        self.read_sfr_crc(&address).await.map(R::from)
    }

    /* Raw SFR Ops (Minimal type checking) */

    async fn read_sfr(&mut self, address: &SFRAddress) -> Result<u32, Error> {
        if self.crc_protection {
            return self.read_sfr_crc(address).await;
        }

        let mut instruction = Instruction(OpCode::READ);
        instruction.set_address(*address as u16);

        let mut buf = [0u8; 4];

        self.spi
            .transaction(&mut [
                Operation::Write(&instruction.into_spi_data()),
                Operation::Read(&mut buf),
            ])
            .await
            .map_err(|_| Error::SPIRead)?;

        Ok(u32::from_le_bytes(buf))
    }

    async fn write_sfr(&mut self, address: &SFRAddress, value: u32) -> Result<(), Error> {
        // The instruction is big-endian on the wire, the register contents little-endian.
        let bytes = value.to_le_bytes();

        if self.crc_protection {
            // For SFR accesses the N field counts bytes (section 4.3.3).
            return self
                .write_crc_transfer(*address as u16, bytes.len() as u8, &bytes)
                .await;
        }

        let mut instruction = Instruction(OpCode::WRITE);
        instruction.set_address(*address as u16);

        self.spi
            .transaction(&mut [
                Operation::Write(&instruction.into_spi_data()),
                Operation::Write(&bytes),
            ])
            .await
            .map_err(|_| Error::SPIWrite)?;

        Ok(())
    }

    /// Writes a single byte of an SFR; `byte_offset` 0 is bits 7:0 (SFR access is
    /// byte-oriented, see datasheet section 4.1).
    async fn write_sfr_byte(
        &mut self,
        address: &SFRAddress,
        byte_offset: u8,
        value: u8,
    ) -> Result<(), Error> {
        let byte_address = *address as u16 + byte_offset as u16;

        if self.crc_protection {
            return self.write_safe_transfer(byte_address, &[value]).await;
        }

        let mut instruction = Instruction(OpCode::WRITE);
        instruction.set_address(byte_address);

        self.spi
            .transaction(&mut [
                Operation::Write(&instruction.into_spi_data()),
                Operation::Write(&[value]),
            ])
            .await
            .map_err(|_| Error::SPIWrite)?;

        Ok(())
    }

    async fn read_sfr_crc(&mut self, address: &SFRAddress) -> Result<u32, Error> {
        let mut buf = [0u8; 4];

        // For SFR accesses the N field counts bytes (section 4.3.2).
        self.read_crc_retrying(*address as u16, buf.len() as u8, &mut buf)
            .await?;

        Ok(u32::from_le_bytes(buf))
    }

    /* CRC-protected SPI instructions (section 4.3) */

    /// Issues `READ_CRC`, retrying on a CRC mismatch. `n_field` is the instruction's N byte:
    /// bytes for SFR accesses, 32-bit words for RAM accesses.
    async fn read_crc_retrying(
        &mut self,
        address: u16,
        n_field: u8,
        data: &mut [u8],
    ) -> Result<(), Error> {
        let mut attempt = 0;

        loop {
            match self.read_crc_transfer(address, n_field, data).await {
                Err(Error::CrcMismatch) if attempt + 1 < CRC_READ_RETRIES => attempt += 1,
                result => return result,
            }
        }
    }

    async fn read_crc_transfer(
        &mut self,
        address: u16,
        n_field: u8,
        data: &mut [u8],
    ) -> Result<(), Error> {
        let mut instruction = Instruction(OpCode::READ_CRC);
        instruction.set_address(address);
        let instruction = instruction.into_spi_data();

        let header = [instruction[0], instruction[1], n_field];
        let mut received_crc = [0u8; 2];

        self.spi
            .transaction(&mut [
                Operation::Write(&header),
                Operation::Read(data),
                Operation::Read(&mut received_crc),
            ])
            .await
            .map_err(|_| Error::SPIRead)?;

        let received_crc = u16::from_be_bytes(received_crc);

        if self.crc_of(&header, data) == received_crc {
            return Ok(());
        }

        // The Linux mcp251xfd driver observed that a READ_CRC of CiTBC whose lowest byte is
        // 0x00 or 0x80 sometimes carries a CRC computed over that byte with bit 7 flipped, and
        // that the flipped data is the correct value.
        if address == SFRAddress::C1TBC as u16
            && data.len() == 4
            && (data[0] & 0xF8 == 0x00 || data[0] & 0xF8 == 0x80)
        {
            data[0] ^= 0x80;

            if self.crc_of(&header, data) == received_crc {
                return Ok(());
            }
        }

        Err(Error::CrcMismatch)
    }

    /// Issues `WRITE_CRC`: the device writes the data, then checks the CRC and raises
    /// `CRC.CRCERRIF` on a mismatch. `n_field` is the instruction's N byte: bytes for SFR
    /// accesses, 32-bit words for RAM accesses.
    async fn write_crc_transfer(
        &mut self,
        address: u16,
        n_field: u8,
        data: &[u8],
    ) -> Result<(), Error> {
        let mut instruction = Instruction(OpCode::WRITE_CRC);
        instruction.set_address(address);
        let instruction = instruction.into_spi_data();

        let header = [instruction[0], instruction[1], n_field];
        let crc = self.crc_of(&header, data).to_be_bytes();

        self.spi
            .transaction(&mut [
                Operation::Write(&header),
                Operation::Write(data),
                Operation::Write(&crc),
            ])
            .await
            .map_err(|_| Error::SPIWrite)
    }

    /// Issues `WRITE_SAFE`, which the device only commits if the CRC matches. Takes exactly
    /// one byte for an SFR (section 4.3.4) or one 4-byte word for RAM (section 4.3.7).
    async fn write_safe_transfer(&mut self, address: u16, data: &[u8]) -> Result<(), Error> {
        debug_assert!(data.len() == 1 || data.len() == 4);

        let mut instruction = Instruction(OpCode::WRITE_SAFE);
        instruction.set_address(address);
        let header = instruction.into_spi_data();

        let crc = self.crc_of(&header, data).to_be_bytes();

        self.spi
            .transaction(&mut [
                Operation::Write(&header),
                Operation::Write(data),
                Operation::Write(&crc),
            ])
            .await
            .map_err(|_| Error::SPIWrite)
    }

    /// CRC-16/CMS over the instruction header and the data, as the device computes it.
    fn crc_of(&self, header: &[u8], data: &[u8]) -> u16 {
        let mut digest = self.crc.digest();
        digest.update(header);
        digest.update(data);
        digest.finalize()
    }

    /* RAM related functions */

    /// Verify SPI connection is working by writing to an available ram location.
    pub async fn verify_spi_communications(&mut self) -> Result<(), ConfigError> {
        let address = 0x400;
        for i in 0..32 {
            let data: u32 = 1 << i;
            self.write_ram(address, &data.to_le_bytes()).await?;

            let mut read_back_buf = [0u8; 4];
            self.read_ram(address, &mut read_back_buf).await?;
            let read_back_value = u32::from_le_bytes(read_back_buf);
            if read_back_value != data {
                return Err(ConfigError::SPIFailedRAMEcho);
            }
        }
        Ok(())
    }

    pub async fn verify_spi_communications_long(&mut self) -> Result<(), ConfigError> {
        let address = 0x400;

        let mut dword_data = [0u32; 32];
        for (i, d) in dword_data.iter_mut().enumerate() {
            *d = 1 << i;
        }

        let mut data = [0u8; 32 * 4];
        for i in 0..32 {
            let bytes = dword_data[i].to_le_bytes();

            data[i * 4] = bytes[0];
            data[i * 4 + 1] = bytes[1];
            data[i * 4 + 2] = bytes[2];
            data[i * 4 + 3] = bytes[3];
        }

        self.write_ram(address, &data).await?;

        let mut read_back_buf = [0u8; 32 * 4];
        self.read_ram(address, &mut read_back_buf).await?;

        if read_back_buf != data {
            return Err(ConfigError::SPIFailedRAMEcho);
        }

        Ok(())
    }

    /// Reads a contiguous range from RAM into the provided buffer
    pub async fn read_ram(&mut self, address: u16, data: &mut [u8]) -> Result<(), Error> {
        is_valid_ram_address(address as u32, data.len())
            .then_some(())
            .ok_or(Error::InvalidRamAddress(address))?;

        if !data.len().is_multiple_of(4) {
            return Err(Error::InvalidReadLength(data.len()));
        }

        if self.crc_protection {
            // For RAM accesses the N field counts 32-bit words (section 4.3.5), so long
            // transfers are split.
            let mut chunk_address = address;
            for chunk in data.chunks_mut(RAM_CRC_MAX_TRANSFER_BYTES) {
                self.read_crc_retrying(chunk_address, (chunk.len() / 4) as u8, chunk)
                    .await?;
                chunk_address += chunk.len() as u16;
            }

            return Ok(());
        }

        let mut instruction = Instruction(OpCode::READ);
        instruction.set_address(address);

        self.spi
            .transaction(&mut [
                Operation::Write(&instruction.into_spi_data()),
                Operation::Read(data),
            ])
            .await
            .map_err(|_| Error::SPIRead)?;

        Ok(())
    }

    /// Writes to a contiguous range in RAM from the provided buffer
    pub async fn write_ram(&mut self, address: u16, data: &[u8]) -> Result<(), Error> {
        is_valid_ram_address(address as u32, data.len())
            .then_some(())
            .ok_or(Error::InvalidRamAddress(address))?;

        if !data.len().is_multiple_of(4) {
            return Err(Error::InvalidWriteLength(data.len()));
        }

        if self.crc_protection {
            // For RAM accesses the N field counts 32-bit words (section 4.3.6), so long
            // transfers are split.
            let mut chunk_address = address;
            for chunk in data.chunks(RAM_CRC_MAX_TRANSFER_BYTES) {
                self.write_crc_transfer(chunk_address, (chunk.len() / 4) as u8, chunk)
                    .await?;
                chunk_address += chunk.len() as u16;
            }

            return Ok(());
        }

        let mut instruction = Instruction(OpCode::WRITE);
        instruction.set_address(address);

        self.spi
            .transaction(&mut [
                Operation::Write(&instruction.into_spi_data()),
                Operation::Write(data),
            ])
            .await
            .map_err(|_| Error::SPIWrite)?;

        Ok(())
    }
}

/* Low level SPI instruction encoding */

bitfield! {
 struct Instruction(u16);
    impl Debug;
    u16;
    pub op_code, set_op_code: 15, 12;
    pub address, set_address: 11, 0;
}

impl Instruction {
    pub fn into_spi_data(self) -> [u8; 2] {
        self.0.to_be_bytes()
    }
}

struct OpCode;

impl OpCode {
    pub const RESET: u16 = 0b0000 << 12;
    pub const READ: u16 = 0b0011 << 12;
    pub const WRITE: u16 = 0b0010 << 12;
    pub const READ_CRC: u16 = 0b1011 << 12;
    pub const WRITE_CRC: u16 = 0b1010 << 12;
    pub const WRITE_SAFE: u16 = 0b1100 << 12;
}

#[cfg(test)]
mod test {
    use super::{round_up_spi_transfer_size, Instruction, OpCode};

    /// The device's CRC is CRC-16 with polynomial 0x8005, initial value 0xFFFF, no reflection
    /// and no final XOR (section 4.3.1), which the catalogue calls CRC-16/CMS.
    #[test]
    fn crc_matches_device_algorithm() {
        let crc = crc::Crc::<u16>::new(&crc::CRC_16_CMS);

        // Bit-serial reference implementation of the datasheet's description.
        fn reference(bytes: &[u8]) -> u16 {
            let mut crc: u16 = 0xFFFF;
            for &byte in bytes {
                crc ^= (byte as u16) << 8;
                for _ in 0..8 {
                    crc = if crc & 0x8000 != 0 {
                        (crc << 1) ^ 0x8005
                    } else {
                        crc << 1
                    };
                }
            }
            crc
        }

        let mut instruction = Instruction(OpCode::READ_CRC);
        instruction.set_address(0x400);
        let instruction = instruction.into_spi_data();
        let frame = [
            instruction[0],
            instruction[1],
            2,
            0xDE,
            0xAD,
            0xBE,
            0xEF,
            1,
            2,
            3,
            4,
        ];

        assert_eq!(crc.checksum(&frame), reference(&frame));
        assert_eq!(crc.checksum(b"123456789"), 0xAEE7);
    }

    #[test]
    fn instruction_encoding() {
        let mut instruction = Instruction(OpCode::WRITE_SAFE);
        instruction.set_address(0xE05);
        assert_eq!(instruction.into_spi_data(), [0xCE, 0x05]);

        let mut instruction = Instruction(OpCode::READ_CRC);
        instruction.set_address(0x400);
        assert_eq!(instruction.into_spi_data(), [0xB4, 0x00]);
    }

    #[test]
    fn test_round_up_spi_transfer_size() {
        assert_eq!(round_up_spi_transfer_size(0), 0);
        assert_eq!(round_up_spi_transfer_size(1), 4);
        assert_eq!(round_up_spi_transfer_size(2), 4);
        assert_eq!(round_up_spi_transfer_size(3), 4);
        assert_eq!(round_up_spi_transfer_size(4), 4);
        assert_eq!(round_up_spi_transfer_size(5), 8);
        assert_eq!(round_up_spi_transfer_size(6), 8);
        assert_eq!(round_up_spi_transfer_size(7), 8);
        assert_eq!(round_up_spi_transfer_size(8), 8);
        assert_eq!(round_up_spi_transfer_size(9), 12);
    }
}
