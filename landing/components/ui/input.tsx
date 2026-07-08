"use client";
import * as React from "react";
import { UploadCloud, Check, AlertCircle } from "lucide-react";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  containerClassName?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = "", label, error, helperText, leftIcon, rightIcon, containerClassName = "", id, type = "text", disabled, ...props }, ref) => {
    const inputId = id || React.useId();
    
    return (
      <div className={`flex flex-col gap-1.5 w-full ${containerClassName}`}>
        {label && (
          <label htmlFor={inputId} className="text-xs font-semibold text-slate-400 uppercase tracking-wider select-none">
            {label}
          </label>
        )}
        <div className="relative flex items-center">
          {leftIcon && (
            <span className="absolute left-4 text-slate-500 pointer-events-none select-none flex items-center justify-center">
              {leftIcon}
            </span>
          )}
          <input
            id={inputId}
            type={type}
            ref={ref}
            disabled={disabled}
            className={`w-full bg-white/5 border rounded-xl py-3 px-4 text-sm text-white placeholder-slate-600 transition-all duration-200 outline-none
              ${leftIcon ? "pl-11" : ""}
              ${rightIcon ? "pr-11" : ""}
              ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-white/10 hover:bg-white/7 focus:border-indigo-500/60 focus:bg-white/8 focus:ring-2 focus:ring-indigo-500/10 focus:ring-offset-1 focus:ring-offset-black"}
              ${error ? "border-red-500/50 focus:border-red-500 focus:ring-red-500/10" : "border-white/5"}
              ${className}
            `}
            {...props}
          />
          {rightIcon && (
            <span className="absolute right-4 text-slate-500 select-none flex items-center justify-center">
              {rightIcon}
            </span>
          )}
        </div>
        {error && (
          <span className="text-xs text-red-400 leading-none mt-1 flex items-center gap-1">
            <AlertCircle className="w-3.5 h-3.5" /> {error}
          </span>
        )}
        {!error && helperText && <span className="text-xs text-slate-500 leading-none mt-1">{helperText}</span>}
      </div>
    );
  }
);
Input.displayName = "Input";

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  helperText?: string;
  containerClassName?: string;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className = "", label, error, helperText, containerClassName = "", id, disabled, ...props }, ref) => {
    const inputId = id || React.useId();
    
    return (
      <div className={`flex flex-col gap-1.5 w-full ${containerClassName}`}>
        {label && (
          <label htmlFor={inputId} className="text-xs font-semibold text-slate-400 uppercase tracking-wider select-none">
            {label}
          </label>
        )}
        <textarea
          id={inputId}
          ref={ref}
          disabled={disabled}
          className={`w-full bg-white/5 border rounded-xl py-3 px-4 text-sm text-white placeholder-slate-600 transition-all duration-200 outline-none resize-none min-h-[100px]
            ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-white/10 hover:bg-white/7 focus:border-indigo-500/60 focus:bg-white/8 focus:ring-2 focus:ring-indigo-500/10 focus:ring-offset-1 focus:ring-offset-black"}
            ${error ? "border-red-500/50 focus:border-red-500 focus:ring-red-500/10" : "border-white/5"}
            ${className}
          `}
          {...props}
        />
        {error && (
          <span className="text-xs text-red-400 leading-none mt-1 flex items-center gap-1">
            <AlertCircle className="w-3.5 h-3.5" /> {error}
          </span>
        )}
        {!error && helperText && <span className="text-xs text-slate-500 leading-none mt-1">{helperText}</span>}
      </div>
    );
  }
);
Textarea.displayName = "Textarea";

// Reusable Checkbox component
export interface CheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  description?: string;
  error?: string;
}

export const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className = "", label, description, error, id, checked, ...props }, ref) => {
    const inputId = id || React.useId();

    return (
      <div className="flex items-start gap-3 w-full cursor-pointer select-none">
        <div className="flex items-center h-5 relative">
          <input
            id={inputId}
            type="checkbox"
            ref={ref}
            checked={checked}
            className="peer sr-only"
            {...props}
          />
          <div className="w-5 h-5 rounded-lg border border-white/10 bg-white/5 transition-all peer-checked:bg-indigo-600 peer-checked:border-indigo-500 flex items-center justify-center peer-focus-visible:ring-2 peer-focus-visible:ring-indigo-500/50 peer-focus-visible:ring-offset-1 peer-focus-visible:ring-offset-black">
            <Check className="w-3.5 h-3.5 text-white opacity-0 peer-checked:opacity-100 transition-opacity" strokeWidth={3} />
          </div>
        </div>
        <div className="flex flex-col gap-0.5">
          <label htmlFor={inputId} className="text-sm text-slate-200 font-semibold cursor-pointer">
            {label}
          </label>
          {description && <p className="text-xs text-slate-500 leading-normal">{description}</p>}
          {error && <span className="text-xs text-red-400 mt-1">{error}</span>}
        </div>
      </div>
    );
  }
);
Checkbox.displayName = "Checkbox";

// Reusable Sliders component for parameters adjustments
export interface SliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
}

export const Slider: React.FC<SliderProps> = ({ label, value, min = 0, max = 100, step = 1, unit = "", className = "", onChange, ...props }) => {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={`flex flex-col gap-2 w-full ${className}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</span>
        <span className="text-xs font-bold text-white font-mono bg-white/5 rounded px-2 py-0.5">
          {value}
          {unit}
        </span>
      </div>
      <div className="relative flex items-center h-4">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
          className="w-full h-1.5 rounded-lg appearance-none bg-white/5 cursor-pointer outline-none focus:ring-1 focus:ring-indigo-500/50"
          style={{
            background: `linear-gradient(to right, #6366f1 0%, #6366f1 ${percentage}%, rgba(255,255,255,0.05) ${percentage}%, rgba(255,255,255,0.05) 100%)`,
          }}
          {...props}
        />
      </div>
    </div>
  );
};
Slider.displayName = "Slider";

// Reusable UploadArea component
export interface UploadAreaProps {
  label?: string;
  description?: string;
  onFileSelect?: (file: File) => void;
  className?: string;
}

export const UploadArea: React.FC<UploadAreaProps> = ({
  label = "Upload file image",
  description = "Drag and drop your sketch or click to select image",
  onFileSelect,
  className = "",
}) => {
  const [dragActive, setDragActive] = React.useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect?.(e.dataTransfer.files[0]);
    }
  };

  return (
    <div
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
      className={`border border-dashed rounded-2xl p-8 text-center flex flex-col items-center justify-center gap-3 transition-all cursor-pointer min-h-[160px]
        ${dragActive ? "border-indigo-500 bg-indigo-500/5" : "border-white/10 hover:border-white/20 bg-white/2 hover:bg-white/3"}
        ${className}
      `}
    >
      <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-slate-400">
        <UploadCloud className="w-5 h-5" />
      </div>
      <div>
        <h4 className="text-xs font-bold text-white uppercase tracking-wider">{label}</h4>
        <p className="text-xs text-slate-500 mt-1 max-w-[200px] leading-relaxed font-light">{description}</p>
      </div>
    </div>
  );
};
UploadArea.displayName = "UploadArea";
