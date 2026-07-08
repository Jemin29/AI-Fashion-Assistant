"use client";
import * as React from "react";
import { motion } from "framer-motion";

export interface ChartDataPoint {
  label: string;
  value: number;
  secondaryValue?: number;
}

export interface BarChartProps {
  data: ChartDataPoint[];
  height?: number;
  color?: "indigo" | "coral" | "teal";
}

export const BarChart: React.FC<BarChartProps> = ({ data, height = 240, color = "indigo" }) => {
  const maxVal = Math.max(...data.map((d) => d.value), 1);
  
  const colors = {
    indigo: "bg-indigo-600 group-hover:bg-indigo-500 shadow-indigo-500/20",
    coral: "bg-brand-coral group-hover:bg-orange-400 shadow-orange-500/20",
    teal: "bg-brand-teal group-hover:bg-teal-400 shadow-teal-500/20",
  };

  return (
    <div className="w-full flex flex-col gap-4">
      <div className="flex items-end gap-3 w-full" style={{ height }}>
        {data.map((item, idx) => {
          const heightPercent = (item.value / maxVal) * 100;
          
          return (
            <div key={idx} className="flex-1 flex flex-col items-center gap-2 group h-full justify-end">
              {/* Tooltip */}
              <div className="opacity-0 group-hover:opacity-100 transition-opacity bg-slate-900 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs font-semibold shadow-md pointer-events-none select-none">
                {item.value.toLocaleString()}
              </div>

              {/* Bar wrapper */}
              <div className="w-full bg-white/5 rounded-t-lg overflow-hidden flex flex-col justify-end h-full">
                <motion.div
                  initial={{ height: 0 }}
                  whileInView={{ height: `${heightPercent}%` }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: idx * 0.05, ease: "easeOut" }}
                  className={`w-full rounded-t-lg transition-colors shadow-lg ${colors[color]}`}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Axis Labels */}
      <div className="flex items-center gap-3 w-full border-t border-white/5 pt-3">
        {data.map((item, idx) => (
          <div key={idx} className="flex-1 text-center text-xs text-slate-500 truncate">
            {item.label}
          </div>
        ))}
      </div>
    </div>
  );
};

export interface LineChartProps {
  data: ChartDataPoint[];
  height?: number;
  strokeColor?: string;
  fillColor?: string;
}

export const LineChart: React.FC<LineChartProps> = ({
  data,
  height = 240,
  strokeColor = "hsl(245, 70%, 62%)",
  fillColor = "rgba(99, 102, 241, 0.1)",
}) => {
  const maxVal = Math.max(...data.map((d) => d.value), 1);
  const width = 500;
  const padding = 40;
  
  const points = data.map((d, i) => {
    const x = padding + (i * (width - padding * 2)) / (data.length - 1 || 1);
    const y = height - padding - (d.value * (height - padding * 2)) / maxVal;
    return { x, y, label: d.label, val: d.value };
  });

  const pathD = points.reduce((acc, p, i) => {
    return i === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`;
  }, "");

  const fillD = points.length > 0 
    ? `${pathD} L ${points[points.length - 1].x} ${height - padding} L ${points[0].x} ${height - padding} Z`
    : "";

  return (
    <div className="w-full flex flex-col gap-3">
      <div className="relative w-full" style={{ height }}>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = padding + ratio * (height - padding * 2);
            return (
              <line
                key={ratio}
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                stroke="rgba(255,255,255,0.03)"
                strokeDasharray="4 4"
              />
            );
          })}

          {/* Area fill */}
          {fillD && <path d={fillD} fill={fillColor} />}

          {/* Path line */}
          {pathD && (
            <motion.path
              d={pathD}
              fill="none"
              stroke={strokeColor}
              strokeWidth="3"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              whileInView={{ pathLength: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            />
          )}

          {/* Data Points */}
          {points.map((p, idx) => (
            <g key={idx} className="group cursor-pointer">
              <motion.circle
                cx={p.x}
                cy={p.y}
                r="5"
                fill={strokeColor}
                stroke="#0f0f15"
                strokeWidth="2"
                initial={{ scale: 0 }}
                whileInView={{ scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.05 }}
                whileHover={{ scale: 1.5 }}
              />
              {/* Point tooltip inside svg */}
              <text
                x={p.x}
                y={p.y - 12}
                textAnchor="middle"
                className="opacity-0 group-hover:opacity-100 transition-opacity duration-150 fill-white text-[10px] font-bold"
              >
                {p.val}
              </text>
            </g>
          ))}
        </svg>
      </div>

      {/* Axis Labels */}
      <div className="flex justify-between px-[32px] border-t border-white/5 pt-3">
        {data.map((item, idx) => (
          <div key={idx} className="text-xs text-slate-500 font-medium">
            {item.label}
          </div>
        ))}
      </div>
    </div>
  );
};
