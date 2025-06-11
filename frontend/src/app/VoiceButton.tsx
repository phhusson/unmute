import React from "react";
import clsx from "clsx";

interface ButtonProps {
  onClick?: () => void;
  children: React.ReactNode;
  kind?: "primary" | "secondary";
}

const VoiceButton: React.FC<ButtonProps> = ({
  onClick = () => {},
  children,
  kind = "primary",
}) => {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "w-full px-2 py-2 bg-gray md:bg-black text-xs lg:text-sm cursor-pointer transition-colors duration-200",
        "overflow-hidden text-nowrap border-1 border-dashed",
        kind === "primary"
          ? "text-green border-green"
          : "text-white border-transparent"
      )}
      // Complex drop shadow easier to do outside of Tailwind
      style={{
        filter: "drop-shadow(0rem 0.2rem 0.15rem var(--darkgray))",
      }}
    >
      {/* The inner span ensures the content overflows in a centered way */}
      <span className="mx-[-100%] text-center">{children}</span>
    </button>
  );
};

export default VoiceButton;
