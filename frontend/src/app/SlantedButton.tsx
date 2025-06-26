import React from "react";
import clsx from "clsx";

const SlantedButton = ({
  onClick = () => {},
  children,
  kind = "primary",
  style,
  extraClasses,
}: {
  onClick?: () => void;
  children: React.ReactNode;
  kind?: "primary" | "secondary" | "disabled";
  style?: React.CSSProperties;
  extraClasses?: string;
}) => {
  const kindToClass = {
    primary: "cursor-pointer after:bg-green text-black after:border-green",
    secondary:
      "cursor-pointer after:bg-darkgray text-white after:border-white after:border-dashed",
    disabled:
      "cursor-not-allowed after:bg-darkgray text-lightgray after:border-lightgray after:border-dashed",
  };

  return (
    <button
      onClick={onClick}
      disabled={kind === "disabled"}
      className={clsx(
        "px-4 py-2 mx-2 z-10 font-medium transition-colors duration-200",
        kindToClass[kind],
        {
          "opacity-50 cursor-not-allowed": kind === "disabled",
          "focus:outline-none focus-visible:outline-4 focus-visible:outline-webkit-focus-ring-color":
            kind !== "disabled",
        },
        extraClasses,
        // Green slanted border
        "relative after:content-[''] after:absolute",
        "after:top-0 after:left-0 after:right-0 after:bottom-0",
        "after:border-2 after:transform after:-skew-x-10 after:-z-10"
      )}
      style={style}
    >
      {children}
    </button>
  );
};

export default SlantedButton;
