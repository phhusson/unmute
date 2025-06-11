import { useEffect, useState } from "react";

const ALLOW_DEV_MODE = false;

const useKeyboardShortcuts = () => {
  // local storage persistence disabled in case random users activate it accidentally
  // useLocalStorage("useDevMode", false)
  const [isDevMode, setIsDevMode] = useState(false);
  // useLocalStorage("showSubtitles", false)
  const [showSubtitles, setShowSubtitles] = useState(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const activeElement = document.activeElement;
      // Don't toggle dev mode if the active element is an input field
      const isInputField =
        activeElement &&
        (activeElement.tagName === "INPUT" ||
          activeElement.tagName === "TEXTAREA" ||
          activeElement.getAttribute("contenteditable") === "true");

      if (
        ALLOW_DEV_MODE &&
        !isInputField &&
        (event.key === "D" || event.key === "d")
      ) {
        setIsDevMode((prev) => !prev);
      }
      if (!isInputField && (event.key === "S" || event.key === "s")) {
        setShowSubtitles((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [setIsDevMode, setShowSubtitles]);

  return { isDevMode, showSubtitles };
};

export default useKeyboardShortcuts;
