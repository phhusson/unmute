import React, { useState, useRef, useEffect } from "react";
import { X } from "lucide-react";
import clsx from "clsx";
import { createPortal } from "react-dom";

interface Position {
  x: number;
  y: number;
}

const Modal = ({
  trigger,
  children,
  forceFullscreen = false,
  closeSignal = 0,
}: {
  trigger: React.ReactNode; // The element that triggers the modal
  children: React.ReactNode; // The content of the modal
  forceFullscreen?: boolean; // Force the modal to be fullscreen
  closeSignal?: number; // Signal to close the modal (increment to trigger)
}) => {
  // Use internal state for controlling open/closed state
  const [isOpen, setIsOpen] = useState(false);

  // Store previous closeSignal value to detect changes
  const prevCloseSignalRef = useRef(closeSignal);

  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const [isMobile, setIsMobile] = useState(false);
  const triggerRef = useRef<HTMLDivElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Use a timeout to prevent flickering
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const isFullscreen = isMobile || forceFullscreen;

  // Effect to handle closeSignal changes
  useEffect(() => {
    // Only run if closeSignal has changed and is not the initial value
    if (closeSignal !== prevCloseSignalRef.current) {
      setIsOpen(false);
      prevCloseSignalRef.current = closeSignal;
    }
  }, [closeSignal]);

  // Detect if device is mobile
  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.matchMedia("(max-width: 768px)").matches);
    };

    checkIsMobile();
    window.addEventListener("resize", checkIsMobile);

    return () => {
      window.removeEventListener("resize", checkIsMobile);
    };
  }, []);

  // Calculate position for the modal
  const calculatePosition = (e: React.MouseEvent) => {
    if (isFullscreen) return;

    const modalWidth = 400; // Default width, adjust as needed
    const modalHeight = 240; // Estimated height, adjust as needed

    let x = e.clientX + 10; // Add a small offset from cursor
    let y = e.clientY + 10; // Add a small offset from cursor

    // Ensure modal stays within viewport
    if (x + modalWidth > window.innerWidth) {
      x = window.innerWidth - modalWidth - 10;
    }

    if (y + modalHeight > window.innerHeight) {
      y = window.innerHeight - modalHeight - 10;
    }

    setPosition({ x, y });
  };

  // Handle mouse enter event
  const handleMouseEnter = (e: React.MouseEvent) => {
    if (!isFullscreen) {
      calculatePosition(e);

      // Clear any existing timeout
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
        hoverTimeoutRef.current = null;
      }

      setIsOpen(true);
    }
  };

  const handleMouseMove = () => {
    // Handle mouse move to update position
    // Disabled so that it doesn't move around
    // if (!isFullscreen && isOpen) {
    //   calculatePosition(e);
    // }
  };

  // Handle mouse leave event with delay to prevent flickering
  const handleMouseLeave = () => {
    if (!isFullscreen) {
      // Set a timeout to close the modal
      hoverTimeoutRef.current = setTimeout(() => {
        setIsOpen(false);
      }, 300);
    }
  };

  // Handle mouse enter on the modal itself
  const handleModalMouseEnter = () => {
    if (!isFullscreen) {
      // Clear the timeout if the mouse enters the modal
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
        hoverTimeoutRef.current = null;
      }
    }
  };

  // Handle click for mobile
  const handleClick = () => {
    if (isFullscreen) {
      setIsOpen(true);
    }
  };

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, []);

  // Close modal when clicking outside on mobile
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        isFullscreen &&
        isOpen &&
        modalRef.current &&
        !modalRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isFullscreen && isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isFullscreen, isOpen]);

  const modalContent = (
    <div
      className={clsx(
        "text-white z-50",
        isFullscreen
          ? "fixed inset-0 bg-black/50 flex items-center justify-center"
          : "fixed"
      )}
      style={
        !isFullscreen
          ? {
              left: `${position.x}px`,
              top: `${position.y}px`,
            }
          : {}
      }
      onMouseEnter={handleModalMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div
        ref={modalRef}
        className={clsx(
          "bg-darkgray border-green shadow-lg text-xs md:text-sm",
          isFullscreen
            ? "max-w-md w-full p-6 max-h-screen overflow-y-auto border-y"
            : "p-4 w-80 border"
        )}
      >
        {isFullscreen && (
          <div className="flex justify-end mb-2 text-white">
            <button onClick={() => setIsOpen(false)} aria-label="Close">
              <X size={24} />
            </button>
          </div>
        )}
        {children}
      </div>
    </div>
  );

  return (
    <div ref={containerRef}>
      {/* Render the trigger element */}
      <div
        ref={triggerRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        className="cursor-pointer"
      >
        {trigger}
      </div>

      {/* Render the modal if open */}
      {isOpen &&
        // Put the modal directly in the body to avoid z-index issues.
        // Otherwise, the circle visualizer appears on top of the modal.
        createPortal(modalContent, document.body)}
    </div>
  );
};

export default Modal;
