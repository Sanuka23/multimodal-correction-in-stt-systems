const STATUS_COLORS = {
  completed: '#4edea3',
  running: '#ffd480',
  failed: '#ffb4ab',
  pending: '#45464d',
  skipped: '#45464d',
}

export default function PipelineConnector({ targetStatus = 'pending' }) {
  const color = STATUS_COLORS[targetStatus] || STATUS_COLORS.pending

  return (
    <svg
      width="40"
      height="24"
      viewBox="0 0 40 24"
      fill="none"
      className="flex-shrink-0 self-center"
    >
      <line
        x1="0"
        y1="12"
        x2="32"
        y2="12"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
      />
      <polygon
        points="32,6 40,12 32,18"
        fill={color}
      />
    </svg>
  )
}
