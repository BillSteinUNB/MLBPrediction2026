import React from "react";

const HELP_TEXT: Record<string, string> = {
  Variant:
    "Which flavor of the model output this run represents, such as base, stacked, or calibrated.",
  Experiment: "The named experiment or run grouping this artifact came from.",
  "Δ":
    "Change versus the previous run in the same lane. Positive is good for ROC AUC and Accuracy. Negative is good for Log Loss and Brier.",
  "ROC AUC":
    "How well the model ranks likely winners above likely losers overall. Higher is better.",
  "Log Loss":
    "How good the predicted probabilities are. Lower is better, and overconfident mistakes are punished hard.",
  Brier: "How close the predicted probabilities were to the actual outcomes. Lower is better.",
  "Brier Score":
    "How close the predicted probabilities were to the actual outcomes. Lower is better.",
  Accuracy:
    "The share of games where the chosen side matched the actual result. Simple, but it ignores confidence.",
  ECE:
    "Expected calibration error. It asks whether 60 percent predictions really win about 60 percent of the time. Lower is better.",
  "Reliability Gap":
    "The worst bucket-level mismatch between predicted probability and actual win rate. Lower is better.",
  "Total Runs": "How many tracked experiment rows are available in the dashboard.",
  "Active Lanes":
    "How many distinct experiment lanes exist when grouped by holdout season, target, and variant.",
  "Best ROC AUC":
    "The strongest run by ranking quality among the currently loaded experiments.",
  "Latest Experiment": "The most recent tracked experiment run in the dataset.",
  Model: "The model family or target-specific model name used for this run.",
  Version: "The artifact version string for the saved model bundle.",
  Target:
    "The column the model is predicting, such as first-five moneyline or first-five run line.",
  Holdout: "The season held out from training and used to judge predictive quality.",
  Timestamp: "When the run artifact was created.",
  "Train rows": "How many rows were used to train this run.",
  "Holdout rows": "How many rows were used to evaluate this run.",
  Features: "The number of model input columns used for this run.",
  Calibration: "The probability adjustment method applied after model training, if any.",
  "F5 Moneyline": "Model probabilities for which team leads after five innings.",
  "F5 Run Line": "Model probabilities for which team covers the first-five run line.",
  "Full Game Market":
    "Public full-game moneyline and run line from live books. This is display-only when first-five odds are not posted yet.",
  "Recommended Bet":
    "The top current decision after edge, odds, and risk rules are applied.",
  Inputs: "Which external inputs were available for this game, including lineups, odds, and weather.",
  Overview: "High-level summary of recent experiment performance and changes.",
  "Live Slate":
    "Dry-run daily model output for a selected date, including every projected game and current pick state.",
  Games: "How many games were returned for the selected date.",
  Picks: "How many games currently have a selected decision after the betting rules are applied.",
  "Projected Lineups":
    "Games where the pipeline found projected lineups instead of only schedule-level placeholders.",
  "Confirmed Lineups":
    "Games where at least one batting order is officially confirmed.",
  "Trend metric": "Which metric to plot over time for each lane.",
  "Lane Explorer": "Browse experiment lanes grouped by holdout season, target, and variant.",
};

export interface TooltipLabelProps {
  label: string;
  helpText?: string;
  as?: keyof React.JSX.IntrinsicElements;
  style?: React.CSSProperties;
}

const wrapperStyle: React.CSSProperties = {
  position: "relative",
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
};

const labelStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  cursor: "help",
  textDecorationLine: "underline",
  textDecorationStyle: "dotted",
  textDecorationColor: "var(--muted)",
  textUnderlineOffset: 3,
};

const iconStyle: React.CSSProperties = {
  width: 16,
  height: 16,
  borderRadius: "50%",
  border: "1px solid var(--border)",
  background: "var(--surface-3)",
  color: "var(--muted)",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 11,
  fontWeight: 700,
  lineHeight: 1,
  flex: "0 0 auto",
};

const bubbleStyle: React.CSSProperties = {
  position: "absolute",
  top: "calc(100% + 8px)",
  left: 0,
  zIndex: 40,
  width: 260,
  maxWidth: "min(32rem, 80vw)",
  padding: "10px 12px",
  borderRadius: 10,
  background: "rgba(7, 11, 20, 0.96)",
  border: "1px solid var(--border)",
  color: "var(--text-h)",
  boxShadow: "0 12px 30px rgba(0, 0, 0, 0.35)",
  fontSize: 12,
  fontWeight: 400,
  lineHeight: 1.45,
  textTransform: "none",
  letterSpacing: "normal",
  pointerEvents: "none",
  whiteSpace: "normal",
};

export const TooltipLabel: React.FC<TooltipLabelProps> = ({
  label,
  helpText,
  as = "span",
  style,
}) => {
  const [open, setOpen] = React.useState(false);
  const text = helpText ?? HELP_TEXT[label];
  const Component = as;

  if (!text) {
    return <Component style={style}>{label}</Component>;
  }

  return (
    <span
      style={wrapperStyle}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <Component style={{ ...labelStyle, ...style }} aria-label={`${label}: ${text}`}>
        <span>{label}</span>
        <span aria-hidden="true" style={iconStyle}>
          ?
        </span>
      </Component>
      {open ? (
        <span role="tooltip" style={bubbleStyle}>
          {text}
        </span>
      ) : null}
    </span>
  );
};

export default TooltipLabel;
