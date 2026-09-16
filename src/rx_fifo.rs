//! Bulk reading of receive FIFOs.
//!
//! [`crate::MCP2518FD::rx_fifo_get_next`] re-reads the FIFO's control, status and user-address
//! registers for every message. [`RxFifoReader`] captures the FIFO geometry once, tracks the
//! tail index in software and lets [`crate::MCP2518FD::rx_fifo_fetch`] pull every pending
//! message with one status read and one RAM read, the way the Linux `mcp251xfd` driver does.

use crate::memory::controller::fifo::FifoNumber;
use crate::message::rx::{RxHeader, RxMessage};
use crate::message::{len_for_dlc, HEADER_SIZE_DWORDS};

const HEADER_LEN: usize = HEADER_SIZE_DWORDS * 4;
const TIMESTAMP_LEN: usize = 4;

/// Geometry and software tail of a receive FIFO, for bulk reads.
///
/// Create with [`crate::MCP2518FD::rx_fifo_reader`], fill a buffer with
/// [`crate::MCP2518FD::rx_fifo_fetch`], then parse the objects with [`RxFifoReader::message`].
///
/// The reader mirrors the controller's tail index, so it must be the only consumer of its FIFO:
/// do not mix it with `rx_fifo_get_next` on the same FIFO, and create a new reader after the
/// FIFO is reconfigured or reset.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RxFifoReader {
    fifo: FifoNumber,
    /// RAM address of message object index 0.
    base_address: u16,
    /// Number of message objects in the FIFO.
    fifo_size: u8,
    /// Payload bytes reserved per object (PLSIZE).
    payload_size: usize,
    with_timestamp: bool,
    /// Index of the next object to read.
    tail: u8,
    /// Timestamp of the last accepted message, for the stale-object check.
    last_timestamp: Option<u32>,
}

impl RxFifoReader {
    pub(crate) fn new(
        fifo: FifoNumber,
        base_address: u16,
        fifo_size: u8,
        payload_size: usize,
        with_timestamp: bool,
        tail: u8,
    ) -> Self {
        Self {
            fifo,
            base_address,
            fifo_size,
            payload_size,
            with_timestamp,
            tail,
            last_timestamp: None,
        }
    }

    pub fn fifo(&self) -> FifoNumber {
        self.fifo
    }

    /// Size in bytes of one message object as stored in RAM: header, optional timestamp and the
    /// FIFO's full payload area regardless of DLC.
    pub fn object_size(&self) -> usize {
        HEADER_LEN
            + if self.with_timestamp {
                TIMESTAMP_LEN
            } else {
                0
            }
            + self.payload_size
    }

    /// Number of message objects in the FIFO.
    pub fn fifo_size(&self) -> u8 {
        self.fifo_size
    }

    /// Whether the objects carry a receive timestamp, which enables the stale-object check in
    /// [`crate::MCP2518FD::rx_fifo_fetch`].
    pub fn has_timestamps(&self) -> bool {
        self.with_timestamp
    }

    /// Number of message objects a buffer of `len` bytes can hold.
    pub fn capacity(&self, len: usize) -> usize {
        len / self.object_size()
    }

    /// Parses object `index` from a buffer filled by [`crate::MCP2518FD::rx_fifo_fetch`].
    ///
    /// Returns `None` if the buffer does not contain that object.
    pub fn message(&self, buf: &[u8], index: usize) -> Option<RxMessage> {
        let object = buf.get(index * self.object_size()..(index + 1) * self.object_size())?;

        let header = RxHeader([
            u32::from_le_bytes(object[0..4].try_into().unwrap()),
            u32::from_le_bytes(object[4..8].try_into().unwrap()),
        ]);

        let timestamp = self
            .with_timestamp
            .then(|| u32::from_le_bytes(object[8..12].try_into().unwrap()));

        let data_offset = HEADER_LEN
            + if self.with_timestamp {
                TIMESTAMP_LEN
            } else {
                0
            };
        // A DLC larger than the FIFO's payload size (DLCMM) leaves the excess bytes unavailable.
        let data_len = len_for_dlc(header.dlc(), header.fdf())
            .unwrap_or(0)
            .min(self.payload_size);

        RxMessage::new(
            header,
            timestamp,
            &object[data_offset..data_offset + data_len],
        )
    }

    /// Timestamp of object `index` in a fetched buffer, if timestamps are enabled.
    pub(crate) fn timestamp_of(&self, buf: &[u8], index: usize) -> Option<u32> {
        if !self.with_timestamp {
            return None;
        }

        let offset = index * self.object_size() + HEADER_LEN;
        buf.get(offset..offset + TIMESTAMP_LEN)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// RAM address of object `index`.
    pub(crate) fn address_of(&self, index: u8) -> u16 {
        self.base_address + index as u16 * self.object_size() as u16
    }

    pub(crate) fn tail(&self) -> u8 {
        self.tail
    }

    /// Number of pending objects given the controller's head index and full flag.
    pub(crate) fn pending(&self, head: u8, full: bool) -> u8 {
        if full {
            self.fifo_size
        } else {
            (head + self.fifo_size - self.tail) % self.fifo_size
        }
    }

    /// Objects available for a single contiguous read from the tail.
    pub(crate) fn contiguous(&self, pending: u8) -> u8 {
        pending.min(self.fifo_size - self.tail)
    }

    /// Returns how many of the `count` objects at the start of `buf` are newer than the last
    /// accepted message, stopping at the first stale one, and records the newest timestamp.
    ///
    /// According to erratum DS80000789E item 6 the FIFOCI head index can be corrupted, which
    /// would make a bulk read run past the head into old objects. Timestamps expose this: an
    /// object older than its predecessor is stale. Without timestamps every object is accepted.
    pub(crate) fn accept_fresh(&mut self, buf: &[u8], count: usize) -> usize {
        if !self.with_timestamp {
            return count;
        }

        let mut accepted = 0;
        for index in 0..count {
            let Some(timestamp) = self.timestamp_of(buf, index) else {
                break;
            };

            if let Some(last) = self.last_timestamp {
                if (timestamp.wrapping_sub(last) as i32) < 0 {
                    break;
                }
            }

            self.last_timestamp = Some(timestamp);
            accepted += 1;
        }

        accepted
    }

    pub(crate) fn advance(&mut self, count: u8) {
        self.tail = (self.tail + count) % self.fifo_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader() -> RxFifoReader {
        RxFifoReader::new(FifoNumber::Fifo1, 0x480, 16, 8, true, 0)
    }

    #[test]
    fn geometry() {
        let reader = reader();
        assert_eq!(reader.object_size(), 20);
        assert_eq!(reader.address_of(0), 0x480);
        assert_eq!(reader.address_of(3), 0x480 + 60);
        assert_eq!(reader.capacity(320), 16);
        assert_eq!(reader.capacity(19), 0);
    }

    #[test]
    fn pending_and_wraparound() {
        let mut reader = reader();
        assert_eq!(reader.pending(0, false), 0);
        assert_eq!(reader.pending(5, false), 5);
        assert_eq!(reader.pending(0, true), 16);

        reader.advance(14);
        assert_eq!(reader.tail(), 14);
        assert_eq!(reader.pending(2, false), 4);
        assert_eq!(reader.contiguous(4), 2);
        reader.advance(2);
        assert_eq!(reader.tail(), 0);
    }

    #[test]
    fn stale_objects_are_rejected() {
        let mut reader = reader();
        let mut buf = [0u8; 20 * 3];
        for (index, timestamp) in [1000u32, 1010, 900].into_iter().enumerate() {
            buf[index * 20 + 8..index * 20 + 12].copy_from_slice(&timestamp.to_le_bytes());
        }

        assert_eq!(reader.accept_fresh(&buf, 3), 2);
        assert_eq!(reader.last_timestamp, Some(1010));

        // Wraparound of the 32-bit counter is not stale.
        let mut reader = RxFifoReader::new(FifoNumber::Fifo1, 0x480, 16, 8, true, 0);
        buf[8..12].copy_from_slice(&(u32::MAX - 10).to_le_bytes());
        buf[28..32].copy_from_slice(&5u32.to_le_bytes());
        assert_eq!(reader.accept_fresh(&buf, 2), 2);
        assert_eq!(reader.last_timestamp, Some(5));
    }

    #[test]
    fn without_timestamps_everything_is_fresh() {
        let mut reader = RxFifoReader::new(FifoNumber::Fifo1, 0x480, 16, 8, false, 0);
        assert_eq!(reader.object_size(), 16);
        assert_eq!(reader.accept_fresh(&[0u8; 48], 3), 3);
    }

    #[test]
    fn parses_messages() {
        let reader = reader();
        let mut buf = [0u8; 20 * 2];
        // Object 1: SID 0x123, DLC 3, timestamp 42, data 1 2 3.
        let t0: u32 = 0x123;
        let t1: u32 = 3;
        buf[20..24].copy_from_slice(&t0.to_le_bytes());
        buf[24..28].copy_from_slice(&t1.to_le_bytes());
        buf[28..32].copy_from_slice(&42u32.to_le_bytes());
        buf[32..35].copy_from_slice(&[1, 2, 3]);

        let message = reader.message(&buf, 1).unwrap();
        assert_eq!(message.timestamp(), Some(42));
        assert_eq!(message.data(), &[1, 2, 3]);
        assert!(reader.message(&buf, 2).is_none());
    }
}
