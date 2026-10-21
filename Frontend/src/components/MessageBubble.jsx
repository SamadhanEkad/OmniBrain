
import { Bot, UserRound, AlertCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

export default function MessageBubble({ message }) {
  const isUser = message?.role === "user";
  const isPending = Boolean(message?.pending);
  const hasError = Boolean(message?.error);

  const content =
    typeof message?.content === "string"
      ? message.content
      : "";

  const displayName = isUser ? "You" : "OmniBrain";

  return (
    <article
      className={`message-row ${
        isUser ? "message-row--user" : ""
      } ${isPending ? "message-row--pending" : ""} ${
        hasError ? "message-row--error" : ""
      }`}
      aria-label={`${displayName} message`}
    >
      <span
        className={`message-avatar ${
          isUser ? "message-avatar--user" : ""
        }`}
        aria-hidden="true"
      >
        {isUser ? (
          <UserRound size={16} />
        ) : (
          <Bot size={17} />
        )}
      </span>

      <div className="message-body">
        <div className="message-meta">
          <strong>{displayName}</strong>

          {!isUser && (
            <span className="ai-label">AI</span>
          )}
        </div>

        {isPending ? (
          <p
            className="message-text message-text--pending"
            role="status"
            aria-live="polite"
          >
            <span
              className="spinner spinner--small"
              aria-hidden="true"
            />
            <span>
              {content || "OmniBrain is preparing your answer..."}
            </span>
          </p>
        ) : isUser ? (
          <div className="message-text">
            <p>{content}</p>
          </div>
        ) : (
          <div className="message-text markdown-answer">
            {content ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[
                  [
                    rehypeKatex,
                    {
                      throwOnError: false,
                      strict: "ignore",
                    },
                  ],
                ]}
                components={{
                  a: ({ node, ...props }) => (
                    <a
                      {...props}
                      target="_blank"
                      rel="noreferrer noopener"
                    />
                  ),

                  img: () => null,

                  table: ({ node, ...props }) => (
                    <div className="markdown-table-wrapper">
                      <table {...props} />
                    </div>
                  ),

                  pre: ({ node, ...props }) => (
                    <pre
                      className="markdown-code-block"
                      {...props}
                    />
                  ),

                  code: ({
                    node,
                    className,
                    children,
                    ...props
                  }) => (
                    <code
                      className={className}
                      {...props}
                    >
                      {children}
                    </code>
                  ),
                }}
              >
                {content}
              </ReactMarkdown>
            ) : (
              <p className="message-text">
                No answer content was returned.
              </p>
            )}
          </div>
        )}

        {hasError && (
          <div
            className="message-error"
            role="alert"
          >
            <AlertCircle
              size={16}
              aria-hidden="true"
            />

            <span>
              {typeof message.error === "string"
                ? message.error
                : "Something went wrong while processing this message."}
            </span>

            {typeof message.onRetry === "function" && (
              <button
                className="text-button"
                type="button"
                onClick={message.onRetry}
                disabled={isPending}
              >
                Retry
              </button>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
