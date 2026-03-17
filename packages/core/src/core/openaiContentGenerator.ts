import type {
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensResponse,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Part,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

export class OpenAIContentGenerator implements ContentGenerator {
  constructor(
    private readonly baseUrl: string,
    private readonly modelName: string,
    private readonly apiKey: string,
    private readonly temperature?: number,
  ) {}

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const openaiRequest = this.convertToOpenAIRequest(request);
    
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(openaiRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    return this.convertToGeminiResponse(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const openaiRequest = this.convertToOpenAIRequest(request);
    openaiRequest.stream = true;

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(openaiRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    
    const createGeminiTextResponse = this.createGeminiTextResponse.bind(this);
    const createGeminiToolCallResponse = this.createGeminiToolCallResponse.bind(this);

    async function* generateStream() {
      let buffer = '';
      let currentToolCall: any = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith('data: ') && trimmedLine !== 'data: [DONE]') {
            const dataStr = trimmedLine.slice(6);
            try {
              const data = JSON.parse(dataStr);
              const choice = data.choices?.[0];
              const delta = choice?.delta;

              if (delta?.tool_calls) {
                for (const toolCall of delta.tool_calls) {
                  if (toolCall.function?.name) {
                    if (currentToolCall) {
                      yield createGeminiToolCallResponse(currentToolCall);
                    }
                    currentToolCall = {
                      name: toolCall.function.name,
                      arguments: toolCall.function.arguments || '',
                    };
                  } else if (toolCall.function?.arguments && currentToolCall) {
                    currentToolCall.arguments += toolCall.function.arguments;
                  }
                }
              } else if (delta?.content) {
                yield createGeminiTextResponse(delta.content);
              }

              if (choice?.finish_reason) {
                if (currentToolCall) {
                  yield createGeminiToolCallResponse(currentToolCall);
                  currentToolCall = null;
                }
                yield {
                  candidates: [
                    {
                      content: { role: 'model', parts: [] },
                      finishReason: choice.finish_reason === 'tool_calls' ? 'STOP' : choice.finish_reason.toUpperCase(),
                    }
                  ]
                } as unknown as GenerateContentResponse;
              }
            } catch (e) {
              // Ignore parse errors for incomplete chunks
            }
          }
        }
      }
      
      if (currentToolCall) {
        yield createGeminiToolCallResponse(currentToolCall);
      }
    }

    return generateStream();
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    return { totalTokens: 0 };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    throw new Error('embedContent is not supported for OpenAI compatible models yet.');
  }

  private convertToOpenAIRequest(request: GenerateContentParameters): any {
    const messages: any[] = [];

    if (request.config?.systemInstruction) {
      let systemContent = '';
      if (typeof request.config.systemInstruction === 'string') {
        systemContent = request.config.systemInstruction;
      } else if ((request.config.systemInstruction as any).parts) {
        systemContent = (request.config.systemInstruction as any).parts.map((p: Part) => p.text || '').join('');
      }
      messages.push({ role: 'system', content: systemContent });
    }

    if (request.contents) {
      for (const content of request.contents as any[]) {
        const role = content.role === 'model' ? 'assistant' : 'user';
        
        if (content.parts) {
          const toolCallParts = content.parts.filter((p: Part) => p.functionCall);
          const toolResponseParts = content.parts.filter((p: Part) => p.functionResponse);
          const textParts = content.parts.filter((p: Part) => p.text);

          if (toolCallParts.length > 0) {
            messages.push({
              role: 'assistant',
              content: textParts.map((p: Part) => p.text).join('') || null,
              tool_calls: toolCallParts.map((p: Part) => ({
                id: p.functionCall?.name || 'call_id',
                type: 'function',
                function: {
                  name: p.functionCall?.name,
                  arguments: JSON.stringify(p.functionCall?.args || {}),
                }
              }))
            });
          } else if (toolResponseParts.length > 0) {
            for (const p of toolResponseParts) {
              messages.push({
                role: 'tool',
                tool_call_id: p.functionResponse?.name || 'call_id',
                name: p.functionResponse?.name,
                content: JSON.stringify(p.functionResponse?.response || {}),
              });
            }
          } else {
            messages.push({
              role,
              content: textParts.map((p: Part) => p.text).join(''),
            });
          }
        }
      }
    }

    const openaiRequest: any = {
      model: this.modelName,
      messages,
    };

    if (this.temperature !== undefined) {
      openaiRequest.temperature = this.temperature;
    }

    if (request.config?.tools) {
      const openaiTools: any[] = [];
      for (const tool of request.config.tools as any[]) {
        if (tool.functionDeclarations) {
          for (const func of tool.functionDeclarations) {
            openaiTools.push({
              type: 'function',
              function: {
                name: func.name,
                description: func.description,
                parameters: func.parameters,
              }
            });
          }
        }
      }
      if (openaiTools.length > 0) {
        openaiRequest.tools = openaiTools;
      }
    }

    return openaiRequest;
  }

  private convertToGeminiResponse(data: any): GenerateContentResponse {
    const choice = data.choices?.[0];
    const message = choice?.message;

    const parts: Part[] = [];
    if (message?.content) {
      parts.push({ text: message.content });
    }

    if (message?.tool_calls) {
      for (const toolCall of message.tool_calls) {
        if (toolCall.type === 'function') {
          parts.push({
            functionCall: {
              name: toolCall.function.name,
              args: JSON.parse(toolCall.function.arguments || '{}'),
            }
          });
        }
      }
    }

    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: choice?.finish_reason === 'tool_calls' ? 'STOP' : choice?.finish_reason?.toUpperCase() || 'STOP',
        }
      ],
      usageMetadata: {
        promptTokenCount: data.usage?.prompt_tokens || 0,
        candidatesTokenCount: data.usage?.completion_tokens || 0,
        totalTokenCount: data.usage?.total_tokens || 0,
      }
    } as unknown as GenerateContentResponse;
  }

  private createGeminiTextResponse(text: string): GenerateContentResponse {
    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [{ text }],
          },
        }
      ],
    } as unknown as GenerateContentResponse;
  }

  private createGeminiToolCallResponse(toolCall: any): GenerateContentResponse {
    let args = {};
    try {
      args = JSON.parse(toolCall.arguments || '{}');
    } catch (e) {
      // If parsing fails, it might be incomplete, but we try our best
    }
    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [{
              functionCall: {
                name: toolCall.name,
                args,
              }
            }],
          },
        }
      ],
    } as unknown as GenerateContentResponse;
  }
}
