import { ChatMessage } from "./chatHistory";
import SingleRoleSubtitles from "./SingleRoleSubtitles";

const Subtitles = ({ chatHistory }: { chatHistory: ChatMessage[] }) => {
  const lastAssistantMessage = chatHistory.findLast(
    (message) => message.role === "assistant" && message.content !== ""
  );
  const lastUserMessage = chatHistory.findLast(
    (message) => message.role === "user" && message.content !== ""
  );

  return (
    <div className="flex flex-col md:flex-row w-full max-w-5xl justify-center">
      <SingleRoleSubtitles
        text={lastAssistantMessage?.content || ""}
        role="assistant"
      />
      <SingleRoleSubtitles text={lastUserMessage?.content || ""} role="user" />
    </div>
  );
};

export default Subtitles;
