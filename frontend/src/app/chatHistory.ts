export type ChatRole = "user" | "assistant" | "system";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

/** If there are multiple messages from the same role in a row, combine them into one message */
export const compressChatHistory = (
  chatHistory: ChatMessage[]
): ChatMessage[] => {
  const compressed: ChatMessage[] = [];
  for (const message of chatHistory) {
    if (
      compressed.length > 0 &&
      compressed[compressed.length - 1].role === message.role
    ) {
      compressed[compressed.length - 1].content += `\n${message.content}`;
    } else {
      compressed.push({ ...message });
    }
  }
  return compressed;
};
