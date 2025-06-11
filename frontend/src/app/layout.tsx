import type { Metadata } from "next";
import "./globals.css";
import { GoogleAnalytics } from "@next/third-parties/google";
import localFont from "next/font/local";

export const metadata: Metadata = {
  title: "Unmute by Kyutai",
  description: "Make LLMs listen and speak.",
};

const satoshi = localFont({
  src: [
    {
      path: "../assets/fonts/Satoshi-Variable.woff2",
      weight: "300 900",
      style: "normal",
    },
    {
      path: "../assets/fonts/Satoshi-VariableItalic.woff2",
      weight: "300 900",
      style: "italic",
    },
  ],
  variable: "--font-satoshi",
  display: "swap",
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={satoshi.className}>
      <head>
        {/* Needed for debugging JSON styling */}
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/pretty-print-json@3.0/dist/css/pretty-print-json.dark-mode.css"
        />
      </head>
      <body>{children}</body>
      {/*
      To debug Google Analytics, add debugMode={true} here and go to the Tag Assistant:
      https://tagassistant.google.com/
      Make sure you don't use an adblocker for localhost, as it will block the GA script.
      */}
      <GoogleAnalytics gaId="G-MLN0BSWF97" />
    </html>
  );
}
