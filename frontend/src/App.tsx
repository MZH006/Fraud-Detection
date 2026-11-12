import { useState } from 'react'

function App() {
  const [form, setForm] = useState({
    customer_id: '',
    terminal_id: '',
    amount: '',
    timestamp: '',
    transaction_id: '',
  })

  const [result, setResult] = useState<null | { transaction_id: string,
    fraud_probability: number,
    is_fraud: boolean
    risk_level: string,
    processing_time: number,
   }>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
      customer_id: Number(form.customer_id),
      terminal_id: Number(form.terminal_id),
      amount: Number(form.amount),
      timestamp: form.timestamp + ':00',
      transaction_id: form.transaction_id || undefined,
    }),
    })
    const data = await response.json()
    setResult(data)
  }
  return (
    <div>
      <h1>Fraud Detection</h1>
      <form onSubmit={handleSubmit}>
        <input type="number" 
        placeholder='Customer ID'
        value={form.customer_id}
        onChange={(e) => setForm({...form, customer_id: e.target.value})}
        />
        <input type="number" 
        placeholder='Terminal ID'
        value={form.terminal_id}
        onChange={(e) => setForm({...form, terminal_id: e.target.value})}
        />
        <input type="number"
        placeholder='Amount'
        value={form.amount}
        onChange={(e) => setForm({...form, amount: (e.target.value)})}
        />
        <input type="datetime-local"
        placeholder='Timestamp'
        value={form.timestamp}
        onChange={(e) => setForm({...form, timestamp: e.target.value})}
        />
        <input type="text"
        placeholder='Transaction ID'
        value={form.transaction_id}
        onChange={(e) => setForm({...form, transaction_id: e.target.value})}
        />
        <button type="submit">Check Transaction</button>
      </form>
      {result && (
        <div>
          <h2>Result</h2>
          <p>Transaction ID: {result.transaction_id}</p>
          <p>Fraud Probability: {result.fraud_probability}</p>
          <p>Is Fraud: {result.is_fraud ? 'Yes' : 'No'}</p>
          <p>Risk Level: {result.risk_level}</p>
          <p>Processing Time: {result.processing_time} ms</p>
        </div>
      )}
    </div>
  )
}

export default App
