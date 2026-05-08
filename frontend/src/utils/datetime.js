/**
 * All user-facing timestamps render in Sri Lanka time (Asia/Colombo, UTC+05:30)
 * regardless of the browser's local timezone. The backend stores UTC.
 *
 * One module, one source of truth — every renderer goes through these helpers.
 */

export const APP_TIMEZONE = 'Asia/Colombo'

const DATE_TIME_FMT = new Intl.DateTimeFormat('en-GB', {
  timeZone: APP_TIMEZONE,
  year: 'numeric',
  month: 'short',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
})

const TIME_FMT = new Intl.DateTimeFormat('en-GB', {
  timeZone: APP_TIMEZONE,
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
})

const DATE_FMT = new Intl.DateTimeFormat('en-GB', {
  timeZone: APP_TIMEZONE,
  year: 'numeric',
  month: 'short',
  day: '2-digit',
})

const FULL_FMT = new Intl.DateTimeFormat('en-GB', {
  timeZone: APP_TIMEZONE,
  weekday: 'short',
  year: 'numeric',
  month: 'short',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
  timeZoneName: 'short',
})

function _toDate(input) {
  if (input == null || input === '') return null
  if (input instanceof Date) return isNaN(input.getTime()) ? null : input
  const d = new Date(input)
  return isNaN(d.getTime()) ? null : d
}

/** "08 May 2026, 14:32" — for transcript pickers and recent lists. */
export function formatDateTime(input) {
  const d = _toDate(input)
  return d ? DATE_TIME_FMT.format(d) : ''
}

/** "14:32:07" — clock time only. */
export function formatTime(input) {
  const d = _toDate(input)
  return d ? TIME_FMT.format(d) : ''
}

/** "08 May 2026" — date only. */
export function formatDate(input) {
  const d = _toDate(input)
  return d ? DATE_FMT.format(d) : ''
}

/** "Fri, 08 May 2026, 14:32:07 GMT+5:30" — full timestamp incl. tz. */
export function formatFullTimestamp(input) {
  const d = _toDate(input)
  return d ? FULL_FMT.format(d) : ''
}

/** "5m ago" / "2h ago" / "3d ago" — relative, timezone-independent. */
export function timeAgo(input) {
  const d = _toDate(input)
  if (!d) return ''
  const diff = Date.now() - d.getTime()
  if (diff < 0) return 'just now'
  const m = Math.floor(diff / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const days = Math.floor(h / 24)
  if (days < 7) return `${days}d ago`
  const weeks = Math.floor(days / 7)
  if (weeks < 4) return `${weeks}w ago`
  return formatDate(d)
}
