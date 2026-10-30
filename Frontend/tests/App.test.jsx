// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../src/App.jsx";
import { api } from "../src/services/api.js";

function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  };
}

function uploadResult(documentId, filename) {
  return {
    uploaded: [
      {
        document_id: documentId,
        filename,
        status: "uploaded",
        page_count: 2,
        size_bytes: 42,
      },
    ],
    failed: [],
  };
}

async function uploadPdf(user, filename = "notes.pdf") {
  const picker = screen.getByLabelText("Choose PDF documents");
  await user.upload(picker, new File(["pdf"], filename, { type: "application/pdf" }));
  await user.click(screen.getByRole("button", { name: "Upload PDF" }));
}

async function ask(user, question) {
  const input = screen.getByLabelText("Ask about your documents");
  await user.type(input, question);
  await user.keyboard("{Enter}");
}

describe("OmniBrain document chat", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("uploads in the composer, sends the current question for that document, and renders the API answer", async () => {
    const user = userEvent.setup();
    let sentQuestion;
    const fetchMock = vi.fn(async (input, options) => {
      const url = new URL(input);
      if (url.pathname === "/health") return jsonResponse({ status: "healthy" });
      if (url.pathname === "/api/documents/upload") {
        return jsonResponse(uploadResult("doc-alpha", "notes.pdf"));
      }
      if (url.pathname === "/api/documents/doc-alpha/answer") {
        sentQuestion = JSON.parse(options.body);
        return jsonResponse({
          answer: `## Key finding

The report identifies a 12% increase.

| Metric | Value |
| --- | --- |
| Increase | 12% |

The equation is $E=mc^2$.

<script>alert("x")</script>`,
          sources: [{
            document_id: "doc-alpha",
            chunk_id: "chunk_1",
            page_number: 2,
            text: "The report identifies a 12% increase.",
          }],
        });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Connected" })).toBeTruthy());
    expect(screen.getAllByRole("button", { name: "Connected" })).toHaveLength(1);
    await uploadPdf(user);
    expect(await screen.findAllByText("notes.pdf")).toHaveLength(2);
    await ask(user, "What is the key finding?");

    await waitFor(() => expect(sentQuestion).toBeDefined());
    expect(sentQuestion).toEqual({ question: "What is the key finding?" });
    await waitFor(() => {
      const answer = document.querySelector(".markdown-answer");
      expect(answer?.textContent || document.querySelector(".message-list")?.textContent)
        .toContain("The report identifies a 12% increase.");
    });
    const renderedAnswer = document.querySelector(".markdown-answer");
    expect(renderedAnswer?.querySelector("h2")?.textContent).toBe("Key finding");
    expect(renderedAnswer?.querySelector("table")).not.toBeNull();
    expect(renderedAnswer?.querySelector(".katex")).not.toBeNull();
    expect(renderedAnswer?.querySelector("script")).toBeNull();
    const requestUrls = fetchMock.mock.calls.map(([url]) => new URL(url));
    const apiOrigin = new URL(api.baseUrl).origin;
    expect(requestUrls.every((url) => url.origin === apiOrigin)).toBe(true);
    expect(requestUrls.map((url) => url.pathname)).toEqual(
      expect.arrayContaining([
        "/health",
        "/api/documents/upload",
        "/api/documents/doc-alpha/answer",
      ]),
    );
  });

  it("shows the health-check error when the backend cannot be reached", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );

    render(<App />);

    expect(await screen.findByRole("button", { name: "Offline · recheck" })).toBeTruthy();
    const healthError = await screen.findByRole("alert");
    expect(healthError.textContent).toContain("Could not reach OmniBrain at");
    expect(healthError.textContent).toContain("CORS settings");
  });

  it("keeps a late response from a previous conversation from replacing the selected document's answer", async () => {
    const user = userEvent.setup();
    let releaseFirstAnswer;
    const firstAnswer = new Promise((resolve) => {
      releaseFirstAnswer = resolve;
    });
    let uploadNumber = 0;
    const answerRequests = [];
    const fetchMock = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/health") return jsonResponse({ status: "healthy" });
      if (url.pathname === "/api/documents/upload") {
        uploadNumber += 1;
        return jsonResponse(uploadResult(`doc-${uploadNumber}`, `file-${uploadNumber}.pdf`));
      }
      if (url.pathname.endsWith("/answer")) {
        answerRequests.push({ path: url.pathname });
        if (url.pathname === "/api/documents/doc-1/answer") return firstAnswer;
        return jsonResponse({
          answer: "Answer from the second PDF.",
          sources: [{ document_id: "doc-2", text: "Second PDF evidence." }],
        });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await uploadPdf(user, "file-1.pdf");
    await ask(user, "First delayed question");
    await waitFor(() => expect(answerRequests).toHaveLength(1));

    await user.click(screen.getByRole("button", { name: /New chat/ }));
    await uploadPdf(user, "file-2.pdf");
    await ask(user, "Question for the second PDF");
    expect(await screen.findByText("Answer from the second PDF.")).toBeTruthy();

    releaseFirstAnswer(jsonResponse({
      answer: "Late answer from the first PDF.",
      sources: [{ document_id: "doc-1", text: "First PDF evidence." }],
    }));
    await waitFor(() => expect(answerRequests).toHaveLength(2));
    await user.click(screen.getByRole("button", { name: "History" }));
    await user.click(screen.getByRole("button", { name: /^Question for the second PDF/ }));
    expect(await screen.findByText("Answer from the second PDF.")).toBeTruthy();
    expect(screen.queryByText("Late answer from the first PDF.")).toBeNull();
    expect(answerRequests.map((request) => request.path)).toEqual([
      "/api/documents/doc-1/answer",
      "/api/documents/doc-2/answer",
    ]);

    await user.click(screen.getByRole("button", { name: "History" }));
    await user.click(screen.getByRole("button", { name: /^First delayed question/ }));
    expect(await screen.findByText("Answer generation stopped.")).toBeTruthy();
    expect(screen.queryByText("Late answer from the first PDF.")).toBeNull();
  });

  it("shows retrieval failures rather than inventing an answer", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/health") return jsonResponse({ status: "healthy" });
      if (url.pathname === "/api/documents/upload") {
        return jsonResponse(uploadResult("doc-empty", "empty.pdf"));
      }
      if (url.pathname === "/api/documents/doc-empty/answer") {
        return jsonResponse({ detail: "No relevant passages were found in this document." }, 404);
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await uploadPdf(user, "empty.pdf");
    await ask(user, "What does this PDF say?");

    expect(await screen.findByText("I couldn’t complete that answer.")).toBeTruthy();
    expect(screen.getByText("No relevant passages were found in this document.")).toBeTruthy();
  });

  it("hides answers whose source metadata identifies a different document", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/health") return jsonResponse({ status: "healthy" });
      if (url.pathname === "/api/documents/upload") {
        return jsonResponse(uploadResult("doc-expected", "expected.pdf"));
      }
      if (url.pathname === "/api/documents/doc-expected/answer") {
        return jsonResponse({
          answer: "This answer belongs to another document.",
          sources: [{ document_id: "doc-other", text: "Foreign evidence." }],
        });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await uploadPdf(user, "expected.pdf");
    await ask(user, "What is in the selected PDF?");

    expect(await screen.findByText(/passages from a PDF outside this search/)).toBeTruthy();
    expect(screen.queryByText("This answer belongs to another document.")).toBeNull();
  });
});
