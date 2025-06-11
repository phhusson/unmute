"use client";
import { useEffect, useState } from "react";
import SquareButton from "./SquareButton";
import { GoogleAnalytics } from "@next/third-parties/google";

export function useCookieConsentState() {
  const [consentGiven, setConsentGiven] = useState<boolean | null>(false);
  const [consentLoaded, setConsentLoaded] = useState<boolean>(false);

  useEffect(() => {
    const consent = localStorage.getItem("cookieConsent");
    setConsentGiven(consent == null ? null : consent === "true");
    setConsentLoaded(true);

    // Listen for localStorage changes (in case consent is given on another tab/page)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === "cookieConsent") {
        setConsentGiven(e.newValue === "true");
      }
    };

    window.addEventListener("storage", handleStorageChange);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
    };
  }, []);

  const setConsent = (to: boolean | null) => {
    if (to != null) {
      localStorage.setItem("cookieConsent", "" + to);
    } else {
      localStorage.removeItem("cookieConsent");
    }
    setConsentGiven(to);
  };

  return {
    consentGiven,
    consentLoaded, // useful to avoid hydration mismatches
    setConsent,
  };
}

export default function CookieConsent() {
  const [showDetails, setShowDetails] = useState(false);
  const { consentGiven, consentLoaded, setConsent } = useCookieConsentState();

  if (!consentLoaded) {
    return null; // Wait until consent state is loaded
  }

  if (consentGiven === true) {
    // To debug Google Analytics, add debugMode={true} here and go to the Tag Assistant:
    // https://tagassistant.google.com/
    // Make sure you don't use an adblocker for localhost, as it will block the GA script.
    return <GoogleAnalytics gaId="G-MLN0BSWF97" />;
  }

  if (consentGiven === false) {
    return null;
  }

  // consent is null, meaning it hasn't been given or declined yet
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gray border-t border-green shadow-lg z-50">
      <div className="max-w-7xl mx-auto p-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex-1">
            <p className="text-sm text-textgray mb-2">
              We use cookies to improve your experience and analyze site usage.{" "}
              {!showDetails && (
                <button
                  onClick={() => setShowDetails(true)}
                  className="text-green underline"
                >
                  Learn more
                </button>
              )}
            </p>

            {showDetails && (
              <div className="mt-3 p-3 bg-darkgray text-sm text-textgray">
                <p className="mb-2">
                  <strong>Analytics Cookies:</strong> We use Google Analytics to
                  understand how visitors interact with our website. This helps
                  us improve our content and user experience.
                </p>
                <button
                  onClick={() => setShowDetails(false)}
                  className="text-green underline"
                >
                  Learn less
                </button>
              </div>
            )}
          </div>

          <div className="flex flex-row gap-2 w-full sm:w-auto justify-center">
            <SquareButton kind="primary" onClick={() => setConsent(true)}>
              Accept
            </SquareButton>
            <SquareButton kind="secondary" onClick={() => setConsent(false)}>
              Decline
            </SquareButton>
          </div>
        </div>
      </div>
    </div>
  );
}
