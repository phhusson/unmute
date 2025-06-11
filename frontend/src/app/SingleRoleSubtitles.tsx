import clsx from "clsx";
import React, { useCallback, useEffect, useRef, useState } from "react";

const SingleRoleSubtitles = ({
  text,
  role,
  nLines = 3,
}: {
  text: string;
  role: "user" | "assistant";
  nLines?: number;
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [displayText, setDisplayText] = useState<string[]>([]);
  const [previousText, setPreviousText] = useState("");

  const updateDisplayText = useCallback(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const containerWidth = container.clientWidth;

    // Create a temporary span to measure text width
    const tempSpan = document.createElement("span");
    tempSpan.style.visibility = "hidden";
    tempSpan.style.position = "absolute";
    tempSpan.style.whiteSpace = "nowrap";
    tempSpan.style.font = window.getComputedStyle(container).font;
    document.body.appendChild(tempSpan);

    const words = text.split(" ");
    const lines: string[] = [];
    let currentLine = "";

    // Build lines word by word
    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      tempSpan.textContent = testLine;

      if (tempSpan.offsetWidth <= containerWidth) {
        currentLine = testLine;
      } else {
        if (currentLine) {
          lines.push(currentLine);
          currentLine = word;
        } else {
          // Word is too long for one line
          lines.push(word);
          currentLine = "";
        }
      }
    }

    // Add the last line if it's not empty
    if (currentLine) {
      lines.push(currentLine);
    }

    // Remove the temporary span
    document.body.removeChild(tempSpan);

    const lastLines = lines.slice(-nLines);
    setDisplayText(lastLines);
  }, [nLines, text]);

  useEffect(() => {
    // If the new text is not a prefix of the old text, reset
    if (!text.startsWith(previousText)) {
      setDisplayText([]);
    }

    setPreviousText(text);

    updateDisplayText();
  }, [previousText, text, updateDisplayText]);

  // Re-calculate when the window resizes
  useEffect(() => {
    const handleResize = () => {
      updateDisplayText();
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [text, updateDisplayText]);

  return (
    // Apply padding from the outside because otherwise we have to take it into
    // account when deciding how to break lines
    <div
      className={clsx(
        "w-full max-w-96 p-2 text-sm lg:text-md xl:text-lg",
        role === "assistant" ? "text-green" : "text-white"
      )}
    >
      <div ref={containerRef} className="h-20">
        {displayText.map((line, index) => (
          <div key={index} className="line whitespace-nowrap">
            {line}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SingleRoleSubtitles;
