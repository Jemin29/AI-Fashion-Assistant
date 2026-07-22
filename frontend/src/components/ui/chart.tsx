"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { cn } from "@/lib/utils";

/* ─── Design Tokens ─────────────────────────────────────────────────────────── */
export const CHART_COLORS = {
  violet:  "oklch(0.62 0.22 275)",
  fuchsia: "oklch(0.72 0.19 315)",
  emerald: "oklch(0.72 0.17 148)",
  blue:    "oklch(0.68 0.18 234)",
  amber:   "oklch(0.80 0.16 72)",
} as const;

export const CHART_COLOR_ARRAY = Object.values(CHART_COLORS);

/* ─── Custom Tooltip ────────────────────────────────────────────────────────── */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ChartTooltip(props: any) {
  const { active, payload, label, formatter } = props as {
    active?: boolean;
    payload?: ReadonlyArray<{ name: string; value: number; color: string }>;
    label?: string;
    formatter?: (value: number) => string;
  };
  if (!active || !payload?.length) return null;

  return (
    <div
      className={cn(
        "rounded-xl border border-border glass-heavy",
        "px-3 py-2.5 shadow-ds-lg text-sm"
      )}
    >
      {label && (
        <p className="text-foreground-muted text-xs mb-2 font-medium">{label}</p>
      )}
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2">
          <span
            className="h-2 w-2 rounded-full shrink-0"
            style={{ background: entry.color }}
          />
          <span className="text-foreground-muted capitalize">{entry.name}:</span>
          <span className="text-foreground font-semibold tabular-nums">
            {formatter ? formatter(entry.value) : entry.value.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ─── Chart Container ───────────────────────────────────────────────────────── */
interface ChartContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  description?: string;
  height?: number;
}

function ChartContainer({
  children,
  title,
  description,
  height = 260,
  className,
  ...props
}: ChartContainerProps) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-border bg-surface-2 p-5",
        className
      )}
      {...props}
    >
      {(title || description) && (
        <div className="mb-4">
          {title && <p className="text-heading-sm text-foreground">{title}</p>}
          {description && (
            <p className="text-body-xs text-foreground-muted mt-0.5">{description}</p>
          )}
        </div>
      )}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          {children as React.ReactElement}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ─── Shared Axis Styles ────────────────────────────────────────────────────── */
const axisStyle = {
  tick: { fill: "oklch(0.52 0.008 260)", fontSize: 11, fontFamily: "Inter" },
  axisLine: { stroke: "oklch(1 0 0 / 0.07)" },
  tickLine: false,
};

const gridStyle = {
  strokeDasharray: "3 3",
  stroke: "oklch(1 0 0 / 0.05)",
  vertical: false,
};

/* ─── Line Chart ────────────────────────────────────────────────────────────── */
interface LineChartProps {
  data: Record<string, string | number>[];
  lines: { key: string; color?: string; name?: string }[];
  xKey: string;
  title?: string;
  description?: string;
  height?: number;
  valueFormatter?: (v: number) => string;
}

function PremiumLineChart({
  data,
  lines,
  xKey,
  title,
  description,
  height,
  valueFormatter,
}: LineChartProps) {
  return (
    <ChartContainer title={title} description={description} height={height}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <defs>
          {lines.map((l, i) => {
            const color = l.color ?? CHART_COLOR_ARRAY[i % CHART_COLOR_ARRAY.length];
            return (
              <linearGradient
                key={l.key}
                id={`line-${l.key}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            );
          })}
        </defs>
        <CartesianGrid {...gridStyle} />
        <XAxis dataKey={xKey} {...axisStyle} dy={8} />
        <YAxis {...axisStyle} dx={-4} />
        <Tooltip
          content={<ChartTooltip formatter={valueFormatter} />}
          cursor={{ stroke: "oklch(1 0 0 / 0.06)", strokeWidth: 1 }}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, color: "oklch(0.52 0.008 260)" }}
        />
        {lines.map((l, i) => {
          const color = l.color ?? CHART_COLOR_ARRAY[i % CHART_COLOR_ARRAY.length];
          return (
            <Line
              key={l.key}
              type="monotone"
              dataKey={l.key}
              name={l.name ?? l.key}
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 5, strokeWidth: 0, fill: color }}
            />
          );
        })}
      </LineChart>
    </ChartContainer>
  );
}

/* ─── Area Chart ────────────────────────────────────────────────────────────── */
interface AreaChartProps {
  data: Record<string, string | number>[];
  areas: { key: string; color?: string; name?: string }[];
  xKey: string;
  title?: string;
  description?: string;
  height?: number;
  stacked?: boolean;
  valueFormatter?: (v: number) => string;
}

function PremiumAreaChart({
  data,
  areas,
  xKey,
  title,
  description,
  height,
  stacked = false,
  valueFormatter,
}: AreaChartProps) {
  return (
    <ChartContainer title={title} description={description} height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <defs>
          {areas.map((a, i) => {
            const color = a.color ?? CHART_COLOR_ARRAY[i % CHART_COLOR_ARRAY.length];
            return (
              <linearGradient
                key={a.key}
                id={`area-${a.key}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop offset="0%" stopColor={color} stopOpacity={0.25} />
                <stop offset="100%" stopColor={color} stopOpacity={0.02} />
              </linearGradient>
            );
          })}
        </defs>
        <CartesianGrid {...gridStyle} />
        <XAxis dataKey={xKey} {...axisStyle} dy={8} />
        <YAxis {...axisStyle} dx={-4} />
        <Tooltip
          content={<ChartTooltip formatter={valueFormatter} />}
          cursor={{ stroke: "oklch(1 0 0 / 0.06)", strokeWidth: 1 }}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "oklch(0.52 0.008 260)" }} />
        {areas.map((a, i) => {
          const color = a.color ?? CHART_COLOR_ARRAY[i % CHART_COLOR_ARRAY.length];
          return (
            <Area
              key={a.key}
              type="monotone"
              dataKey={a.key}
              name={a.name ?? a.key}
              stroke={color}
              strokeWidth={2}
              fill={`url(#area-${a.key})`}
              stackId={stacked ? "1" : undefined}
            />
          );
        })}
      </AreaChart>
    </ChartContainer>
  );
}

/* ─── Bar Chart ─────────────────────────────────────────────────────────────── */
interface BarChartProps {
  data: Record<string, string | number>[];
  bars: { key: string; color?: string; name?: string }[];
  xKey: string;
  title?: string;
  description?: string;
  height?: number;
  horizontal?: boolean;
  valueFormatter?: (v: number) => string;
}

function PremiumBarChart({
  data,
  bars,
  xKey,
  title,
  description,
  height,
  horizontal = false,
  valueFormatter,
}: BarChartProps) {
  const layout = horizontal ? "vertical" : "horizontal";
  return (
    <ChartContainer title={title} description={description} height={height}>
      <BarChart
        data={data}
        layout={layout}
        margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
      >
        <CartesianGrid {...gridStyle} horizontal={!horizontal} vertical={horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" {...axisStyle} dx={-4} />
            <YAxis type="category" dataKey={xKey} {...axisStyle} width={80} />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} {...axisStyle} dy={8} />
            <YAxis {...axisStyle} dx={-4} />
          </>
        )}
        <Tooltip
          content={<ChartTooltip formatter={valueFormatter} />}
          cursor={{ fill: "oklch(1 0 0 / 0.03)" }}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "oklch(0.52 0.008 260)" }} />
        {bars.map((b, i) => {
          const color = b.color ?? CHART_COLOR_ARRAY[i % CHART_COLOR_ARRAY.length];
          return (
            <Bar
              key={b.key}
              dataKey={b.key}
              name={b.name ?? b.key}
              fill={color}
              radius={[4, 4, 0, 0]}
              fillOpacity={0.85}
            />
          );
        })}
      </BarChart>
    </ChartContainer>
  );
}

/* ─── Donut / Pie Chart ─────────────────────────────────────────────────────── */
interface DonutChartProps {
  data: { name: string; value: number; color?: string }[];
  title?: string;
  description?: string;
  height?: number;
  innerRadius?: number;
  centerLabel?: string;
}

function PremiumDonutChart({
  data,
  title,
  description,
  height = 260,
  innerRadius = 70,
  centerLabel,
}: DonutChartProps) {
  const total = data.reduce((s, d) => s + d.value, 0);

  return (
    <ChartContainer title={title} description={description} height={height}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={innerRadius}
          outerRadius={innerRadius + 36}
          paddingAngle={3}
          dataKey="value"
          strokeWidth={0}
        >
          {data.map((entry, i) => (
            <Cell
              key={`cell-${i}`}
              fill={entry.color ?? CHART_COLOR_ARRAY[i % CHART_COLOR_ARRAY.length]}
              opacity={0.88}
            />
          ))}
        </Pie>
        <Tooltip
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          content={(props: any) => (
            <ChartTooltip
              {...props}
              formatter={(v: number) => `${((v / total) * 100).toFixed(1)}%`}
            />
          )}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, color: "oklch(0.52 0.008 260)" }}
          iconType="circle"
          iconSize={8}
        />
        {centerLabel && (
          <text
            x="50%"
            y="50%"
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-foreground text-sm font-semibold"
            style={{ fill: "oklch(0.97 0.004 260)", fontSize: 13, fontWeight: 600 }}
          >
            {centerLabel}
          </text>
        )}
      </PieChart>
    </ChartContainer>
  );
}

export {
  ChartContainer,
  ChartTooltip,
  PremiumLineChart,
  PremiumAreaChart,
  PremiumBarChart,
  PremiumDonutChart,
};
