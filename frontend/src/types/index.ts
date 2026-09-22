export type StepKind = 'agent' | 'tool' | 'result'

/** 一步 Agent 执行记录，对应后端 agent_step / tool_call / tool_result 事件 */
export interface AgentStep {
  kind: StepKind
  text: string
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  steps?: AgentStep[]
}

export interface KnowledgeItem {
  id: string
  content: string
}