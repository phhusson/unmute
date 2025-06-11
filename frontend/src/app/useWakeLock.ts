import { useEffect, useRef } from "react";

const useWakeLock = (shouldPreventSleep: boolean) => {
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);

  useEffect(() => {
    const requestWakeLock = async () => {
      try {
        if ("wakeLock" in navigator && shouldPreventSleep) {
          wakeLockRef.current = await navigator.wakeLock.request("screen");
        }
      } catch (err) {
        console.error("Failed to acquire wake lock:", err);
      }
    };

    const releaseWakeLock = () => {
      if (wakeLockRef.current) {
        wakeLockRef.current.release().catch((err) => {
          console.error("Failed to release wake lock:", err);
        });
        wakeLockRef.current = null;
      }
    };

    if (shouldPreventSleep) {
      requestWakeLock();
    } else {
      releaseWakeLock();
    }

    return () => {
      releaseWakeLock();
    };
  }, [shouldPreventSleep]);
};

export default useWakeLock;
