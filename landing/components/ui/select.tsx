import * as React from "react";
import { ChevronDown, Check } from "lucide-react";

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "onChange"> {
  label?: string;
  options: SelectOption[];
  value: string;
  onChange: (value: string) => void;
  error?: string;
  helperText?: string;
  containerClassName?: string;
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = "", label, options, value, onChange, error, helperText, containerClassName = "", id, disabled, ...props }, ref) => {
    const selectId = id || React.useId();
    
    return (
      <div className={`flex flex-col gap-1.5 w-full ${containerClassName}`}>
        {label && (
          <label htmlFor={selectId} className="text-xs font-semibold text-slate-400 uppercase tracking-wider select-none">
            {label}
          </label>
        )}
        <div className="relative flex items-center">
          <select
            id={selectId}
            ref={ref}
            value={value}
            disabled={disabled}
            onChange={(e) => onChange(e.target.value)}
            className={`w-full bg-white/5 border rounded-xl py-3 pl-4 pr-10 text-sm text-white transition-all duration-200 outline-none appearance-none cursor-pointer
              ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-white/10 hover:bg-white/7 focus:border-indigo-500/60 focus:bg-white/8 focus:ring-2 focus:ring-indigo-500/10"}
              ${error ? "border-red-500 focus:border-red-500 focus:ring-red-500/10" : "border-white/5"}
              ${className}
            `}
            {...props}
          >
            {options.map((opt) => (
              <option key={opt.value} value={opt.value} className="bg-[hsl(225,25%,6%)] text-white">
                {opt.label}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-4 w-4 h-4 text-slate-500 pointer-events-none" />
        </div>
        {error && <span className="text-xs text-red-500 leading-none">{error}</span>}
        {!error && helperText && <span className="text-xs text-slate-500 leading-none">{helperText}</span>}
      </div>
    );
  }
);
Select.displayName = "Select";
