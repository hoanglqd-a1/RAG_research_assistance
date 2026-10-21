const $ = (selector) => document.querySelector(selector);
let conversationId = null;
let busy = false;
const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};
function notice(text, error = false) {
  $('#notice').textContent = text;
  $('#notice').className = error ? 'error' : '';
}
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Dữ liệu không hợp lệ. Kiểm tra câu hỏi và số đoạn (1–20).');
  return data;
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll('button, input, select, textarea').forEach(el => { el.disabled = value; });
  $('#send').textContent = value ? 'Đang xử lý…' : 'Gửi câu hỏi ↗';
  $('#top-k').disabled = value || $('#mode').value === 'chat';
}
async function refreshDocuments() {
  const documents = await request('/api/documents');
  $('#count').textContent = documents.length;
  $('#documents').replaceChildren(...documents.map(doc => {
    const item = node('li', doc.filename);
    item.append(node('small', `${doc.chunks} đoạn · ${doc.pages ? doc.pages + ' trang có văn bản' : 'Văn bản'}`));
    return item;
  }));
}
function message(role, text) {
  $('.welcome')?.remove();
  const item = node('article', undefined, `message ${role}`);
  item.append(node('h3', role === 'user' ? 'Bạn' : 'Paper Assistant'), node('div', text, 'answer'));
  $('#messages').append(item);
  return item;
}
$('#upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  const files = [...$('#files').files];
  if (!files.length) return notice('Chọn ít nhất một file để tải lên.', true);
  if (files.some(file => file.size > 20 * 1024 * 1024)) return notice('Dung lượng tối đa là 20 MB mỗi file.', true);
  setBusy(true);
  try {
    for (const file of files) {
      notice(`Đang lập chỉ mục ${file.name}… Lần đầu có thể cần tải embedding model.`);
      const body = new FormData(); body.append('file', file);
      await request('/api/documents', {method: 'POST', body});
    }
    $('#files').value = '';
    notice('Tài liệu đã sẵn sàng. Bạn có thể bắt đầu hỏi.');
  } catch (error) { notice(error.message, true); }
  finally {
    try { await refreshDocuments(); } catch (error) { notice(error.message, true); }
    setBusy(false);
  }
});
$('#question-form').addEventListener('submit', async event => {
  event.preventDefault();
  const text = $('#question').value.trim();
  if (!text || busy) return;
  const mode = $('#mode').value;
  setBusy(true);
  const userMessage = message('user', text);
  notice(mode === 'chat' ? 'Agent đang đọc tài liệu và xử lý câu hỏi…' : 'Đang tìm đoạn liên quan…');
  try {
    const data = await request(`/api/${mode}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: text, conversation_id: conversationId, top_k: Number($('#top-k').value)})});
    if (mode === 'chat') {
      conversationId = data.conversation_id;
      const item = message('assistant', data.answer);
      for (const source of data.sources) item.append(node('span', `${source.filename}${source.page ? ' · trang ' + source.page : ''}`, 'source'));
      if (data.tool_calls.length) {
        const details = node('details');
        details.append(node('summary', `${data.tool_calls.length} lần gọi tool · ${data.llm_calls} lần gọi model`), node('pre', JSON.stringify(data.tool_calls, null, 2)));
        item.append(details);
      }
      if (data.stopped_due_to_limit) item.append(node('p', 'Agent đã chạm giới hạn bước. Hãy thử câu hỏi cụ thể hơn.'));
    } else {
      const item = message('assistant', data.results.length ? `Tìm thấy ${data.results.length} đoạn liên quan` : 'Không tìm thấy đoạn phù hợp.');
      for (const result of data.results) {
        item.append(node('span', `${result.chunk.filename}${result.chunk.page ? ' · trang ' + result.chunk.page : ''} · điểm ${result.score.toFixed(4)}`, 'source'), node('div', result.chunk.text, 'excerpt'));
      }
    }
    $('#question').value = '';
    notice('');
    $('#messages').lastElementChild?.scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) { userMessage.remove(); notice(error.message, true); }
  finally { setBusy(false); $('#question').focus(); }
});
$('#new-chat').addEventListener('click', () => { conversationId = null; $('#messages').replaceChildren(); notice('Đã bắt đầu hội thoại mới. Thư viện paper vẫn sẵn sàng.'); });
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { $('#question').value = button.dataset.prompt; $('#question').focus(); }));
$('#mode').addEventListener('change', () => { $('#top-k').disabled = $('#mode').value === 'chat'; });
$('#top-k').disabled = true;
Promise.all([request('/api/health').then(data => { $('#system-status').textContent = `● Backend sẵn sàng · ${data.retrieval_mode} · ${data.model}`; }), refreshDocuments()]).catch(error => notice(`Không thể kết nối backend: ${error.message}`, true));
