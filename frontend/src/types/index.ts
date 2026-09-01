export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
}

export interface KnowledgeItem {
  id: string
  content: string
}