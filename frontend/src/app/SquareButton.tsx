import React from "react";
import clsx from "clsx";

const SquareButton = ({
  onClick = () => {},
  children,
  kind = "primary",
  extraClasses,
}: {
  onClick?: () => void;
  children: React.ReactNode;
  kind?: "primary" | "primaryOff" | "secondary";
  extraClasses?: string;
}) => {
  const kindToClass = {
    primary: "text-green border-green",
    primaryOff: "text-white border-white",
    secondary: "text-white border-transparent",
  };

  return (
    <button
      onClick={onClick}
      className={clsx(
        "px-2 py-2 bg-black text-xs lg:text-sm cursor-pointer transition-colors duration-200",
        "overflow-hidden text-nowrap border-1 border-dashed",
        kindToClass[kind],
        extraClasses
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

export default SquareButton;
