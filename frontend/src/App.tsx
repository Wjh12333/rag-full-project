import { useState, useRef } from 'react';
import { Message } from './types';

const BASE_API = "http://127.0.0.1:8000";

export default function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

// 文件上传
const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
  const file = e.target.files?.[0];
  if (!file) return;
  setUploadLoading(true);
  try {
    const formData = new FormData();
    // 后端要求字段名是 file，直接把浏览器File对象塞进去，不要自己读text
    formData.append("file", file);

    const resp = await fetch(`${BASE_API}/upload`, {
      method: "POST",
      body: formData
    });
    const resJson = await resp.json();
    if (resp.ok && resJson.code ===0) {
      setMessages(prev => [...prev, { role: "assistant", content: `✅文件【${file.name}】上传向量化完成，共${resJson.chunk_count}个文档块，可以开始提问文档内容` }])
    } else {
      setMessages(prev => [...prev, { role: "assistant", content: `❌上传失败: ${resJson.msg}` }])
    }
  } catch (err) {
    console.error(err);
    setMessages(prev => [...prev, { role: "assistant", content: "上传异常，请确认后端已启动，并且开启CORS跨域" }])
  } finally {
    setUploadLoading(false);
    if(fileInputRef.current) fileInputRef.current.value = "";
  }
}

  // 对话流式接口 /agent_stream
  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMsg: Message = { role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMsg]);
    const queryText = input.trim();
    setInput('');
    setLoading(true);

    abortRef.current = new AbortController();
    try {
      const res = await fetch(`${BASE_API}/agent_stream?user_query=${encodeURIComponent(queryText)}&max_loop=3`, {
        method: 'GET',
        signal: abortRef.current.signal
      });
      if (!res.ok) throw new Error(`http ${res.status}`);
      const reader = res.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let assistantContent = '';
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        assistantContent += chunk;
        setMessages(prev => {
          const copy = [...prev];
          copy[copy.length-1] = { role: 'assistant', content: assistantContent };
          return copy;
        });
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: "请求出错，请确认后端127.0.0.1:8000运行，开启CORS" }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth:800, margin:'20px auto', padding:'0 16px' }}>
      <h2>RAG-Agent 对话助手</h2>

      <div style={{marginBottom:12}}>
        <input ref={fileInputRef} type="file" accept=".txt" onChange={handleFileUpload} style={{display:"none"}}/>
        <button onClick={()=>fileInputRef.current?.click()} disabled={uploadLoading} style={{padding:"6px 12px"}}>
          {uploadLoading ? "文档向量化中..." : "上传txt文档"}
        </button>
        <span style={{marginLeft:10,color:"#666"}}>仅支持txt文本文件</span>
      </div>

      <div style={{ border:'1px solid #ccc', minHeight:400, padding:12, marginBottom:12, borderRadius:6 }}>
        {messages.map((m,i)=>(
          <div key={i} style={{ margin:'8px 0' }}>
            <div style={{ fontWeight:'bold' }}>{m.role==='user'?'用户':'AI'}</div>
            <div style={{ whiteSpace:'pre-wrap' }}>{m.content}</div>
          </div>
        ))}
      </div>
      <textarea
        value={input}
        onChange={(e)=>setInput(e.target.value)}
        placeholder="输入你的问题..."
        rows={4}
        style={{ width:'100%', padding:8 }}
      />
      <div style={{ marginTop:8 }}>
        <button onClick={handleSend} disabled={loading} style={{ padding:'8px 16px' }}>
          {loading ? "生成中..." : "发送"}
        </button>
      </div>
    </div>
  )
}
