'use strict';
const state = {runs: [], symbol: 'BTCUSDT', run: null, trades: [], page: 0, busy: false};
const $ = id => document.getElementById(id);
const names = {BTCUSDT: 'بیت‌کوین', ETHUSDT: 'اتریوم'};
const symbols = {BTCUSDT: 'BTC', ETHUSDT: 'ETH'};
const number = value => new Intl.NumberFormat('en-US', {maximumFractionDigits: 2}).format(value);
const percent = value => value == null ? '—' : `${(value * 100).toFixed(2)}%`;
const signed = value => value == null ? '—' : `${value > 0 ? '+' : ''}${percent(value)}`;
const tone = value => value > 0 ? 'positive' : value < 0 ? 'negative' : '';
const date = value => new Intl.DateTimeFormat('fa-IR', {timeZone:'Asia/Tehran', year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}).format(new Date(value));
const safe = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

async function get(url) {
  const response = await fetch(url, {cache:'no-store', signal:AbortSignal.timeout(15000)});
  if (!response.ok) throw new Error(`خطا در دریافت داده (${response.status})`);
  return response.json();
}

function assets() {
  $('assets').innerHTML = Object.keys(names).map(symbol => {
    const run = state.runs.find(item => item.symbol === symbol);
    return `<button class="asset-card ${symbol === state.symbol ? 'current' : ''}" data-asset="${symbol}" aria-pressed="${symbol === state.symbol}"><div class="asset-name"><span class="coin-icon ${symbol === 'ETHUSDT' ? 'eth' : ''}">${symbol === 'BTCUSDT' ? '₿' : 'Ξ'}</span><div><strong>${names[symbol]}</strong><small>${symbol} · آخرین اجرای آموزش</small></div></div><div class="asset-return ${tone(run?.total_return_compounded)}">${run ? signed(run.total_return_compounded) : '—'}<small>${run ? `${run.trade_count} معامله · برد ${percent(run.win_rate)}` : 'بدون نتیجهٔ آموزش'}</small></div></button>`;
  }).join('');
  document.querySelectorAll('[data-asset]').forEach(button => button.addEventListener('click', () => selectSymbol(button.dataset.asset)));
}

async function refresh() {
  if (state.busy) return;
  state.busy = true;
  controls(true);
  $('error').hidden = true;
  try {
    await refreshPortfolio();
    await refreshSignals();
    const runs = await get('/runs');
    const matching = runs.filter(run => run.symbol === state.symbol);
    const run = matching.find(item => item.run_id === state.run?.run_id) || matching[0] || null;
    const trades = run && run.trade_count > 0 ? await get(`/trades/${run.run_id}`) : [];
    state.runs = runs;
    state.run = run;
    state.trades = trades;
    state.page = 0;
    render();
    $('status').textContent = `آخرین دریافت: ${new Intl.DateTimeFormat('fa-IR', {timeZone:'Asia/Tehran',hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(new Date())} · تازه‌سازی نتایج ذخیره‌شده هر ۶۰ ثانیه`;
  } catch (error) { showError(error); }
  finally {state.busy = false; controls(false);}
}

async function refreshPortfolio() {
  try {
    const wallet = await get('/portfolio');
    if (!wallet.available) { $('paper-status').textContent = wallet.reason; return; }
    const stale = Date.now() - wallet.updated_at > 30 * 60000;
    $('paper-status').textContent = `${stale ? 'داده قدیمی · ' : ''}آخرین اجرا: ${date(wallet.updated_at)} · ${wallet.universe.length} ارز · تلگرام: ${wallet.report_status} · ${wallet.halted ? 'خرید متوقف است' : 'خرید فعال است'} · ${Object.keys(wallet.market_errors).length} خطای بازار`;
    const model = wallet.model_metrics;
    $('paper-training').textContent = `آموزش: ${date(Date.parse(wallet.model_trained_at))} · ${number(model.train_rows)} نمونه آموزش · ${number(model.validation_rows)} اعتبارسنجی · ${number(model.test_rows)} آزمون · Brier: ${model.test_brier.toFixed(3)} · AUC: ${model.test_auc == null ? '—' : model.test_auc.toFixed(3)} · این معیارها بازده کیف پول نیستند.`;
    const threshold = wallet.entry_threshold ?? model.entry_threshold;
    if (Number.isFinite(threshold)) $('paper-training').textContent += ` آستانه خرید: ${threshold.toFixed(3)}.`;
    const policyStatus = model.threshold_selection?.status;
    $('paper-training').textContent += policyStatus === 'validated'
      ? ' آستانه از معاملات اعتبارسنجی انتخاب شده است.'
      : policyStatus === 'no_validated_threshold'
        ? ' هیچ آستانه‌ای معیار اعتبارسنجی را نگذرانده؛ آستانه ثابت حفظ شده است.'
        : ' مدل قدیمی است؛ برای بررسی آستانه دوباره آموزش دهید.';
    $('paper-universe').innerHTML = wallet.universe.map(symbol => {
      const score = wallet.signal_scores?.[symbol];
      return `<span class="pill">${safe(symbol)} · ${safe(wallet.signals[symbol] || wallet.market_errors[symbol] || '—')}${Number.isFinite(score) ? ` · امتیاز ${score.toFixed(3)}` : ''}</span>`;
    }).join(' ');
    const metrics = [['ارزش خالص', wallet.equity], ['موجودی نقد', wallet.cash], ['سود / زیان کل', wallet.total_pnl], ['کارمزد پرداخت‌شده', wallet.fees_paid]];
    $('paper-metrics').innerHTML = metrics.map(([label,value]) => `<article class="metric"><div class="metric-label">${label}</div><div class="metric-number">${value.toFixed(4)}</div><div class="metric-detail">USDT مجازی</div></article>`).join('');
    const points = [{time:wallet.equity_history[0]?.time || wallet.updated_at, equity:wallet.initial_balance}, ...wallet.equity_history];
    const frame = chartFrame('paper-chart', points.map(point => point.equity), '#57d6ac', '');
    const path = points.map((point,index) => `${index ? 'L' : 'M'} ${frame.x(index)} ${frame.y(point.equity)}`).join(' ');
    $('paper-chart').innerHTML = frame.markup + `<path d="${path}" stroke="#57d6ac" stroke-width="2.4" fill="none"/>` + points.map((point,index) => `<circle cx="${frame.x(index)}" cy="${frame.y(point.equity)}" r="3" fill="#57d6ac"><title>${date(point.time)} · ${point.equity.toFixed(4)} USDT</title></circle>`).join('') + '</svg>';
    $('paper-positions').innerHTML = '<h3>موقعیت‌های باز</h3><table><thead><tr><th>ارز</th><th>مقدار</th><th>هزینه ورود</th><th>حد ضرر</th><th>حد سود</th></tr></thead><tbody>' + Object.values(wallet.positions).map(position => `<tr><td>${safe(position.symbol)}</td><td>${position.quantity}</td><td>${position.cost.toFixed(4)}</td><td>${position.stop.toPrecision(6)}</td><td>${position.target.toPrecision(6)}</td></tr>`).join('') + '</tbody></table>';
    $('paper-trades').innerHTML = '<h3>آخرین خرید و فروش‌ها</h3><table><thead><tr><th>زمان</th><th>ارز</th><th>عملیات</th><th>مبلغ</th><th>سود / زیان</th><th>علت</th></tr></thead><tbody>' + wallet.trade_history.slice(-20).reverse().map(trade => `<tr><td>${date(trade.time)}</td><td>${safe(trade.symbol)}</td><td>${safe(trade.side)}</td><td>${trade.notional.toFixed(4)}</td><td>${trade.pnl == null ? '—' : trade.pnl.toFixed(4)}</td><td>${safe(trade.reason)}</td></tr>`).join('') + '</tbody></table>';
  } catch (_) { $('paper-status').textContent = 'دریافت کیف پول ناموفق بود؛ داده‌های نمایان ممکن است قدیمی باشند.'; }
}

async function refreshSignals() {
  try {
    const signals = await get('/signals');
    const deliveryNames = {preview:'پیش‌نمایش · بدون ارسال',sent:'ارسال به تلگرام تأیید شد',duplicate_suppressed:'ارسال تکراری متوقف شد',not_required:'امتیاز زیر آستانه · بدون ارسال',unknown_or_failed:'ارسال ناموفق یا نامشخص',not_requested:'ارسال درخواست نشده'};
    $('live-signals').innerHTML = Object.keys(names).map(symbol => {
      const signal = signals.find(item => item.symbol === symbol);
      if (!signal) return `<article class="live-card"><strong>${names[symbol]}</strong><p>هنوز ارزیابی زنده‌ای ثبت نشده است.</p></article>`;
      if (signal.action === 'ERROR') return `<article class="live-card"><strong>${names[symbol]}</strong><p class="negative">ارزیابی ناموفق</p><p dir="ltr">${safe(signal.error)}</p><small>زمان بررسی: ${date(Date.parse(signal.evaluated_at))}</small></article>`;
      if (signal.action === 'WAITING') return `<article class="live-card"><strong>${names[symbol]}</strong><p>در انتظار کندل جدید پس از آموزش</p><small>زمان بررسی: ${date(Date.parse(signal.evaluated_at))}</small></article>`;
      const stale = Date.now() - signal.candle_close_time > 30 * 60 * 1000;
      return `<article class="live-card"><div class="live-card-heading"><strong>${names[symbol]}</strong><span class="${signal.action === 'BUY' && !stale ? 'positive' : ''}">${stale ? 'نتیجهٔ قدیمی' : signal.action === 'BUY' ? 'پیشنهاد پژوهشی خرید' : 'بدون سیگنال خرید'}</span></div><div class="live-score">${signal.model_score.toFixed(3)}<small>امتیاز مدل · آستانه ${signal.threshold}</small></div><div class="live-prices"><span>قیمت مرجع <b>${number(signal.reference_price)}</b></span><span>حد ضرر مرجع <b>${number(signal.reference_stop)}</b></span><span>حد سود مرجع <b>${number(signal.reference_target)}</b></span></div><small>بسته‌شدن کندل: ${date(signal.candle_close_time)}</small><small>زمان ارزیابی: ${date(Date.parse(signal.evaluated_at))}</small><p class="delivery">${safe(deliveryNames[signal.delivery_status] || signal.delivery_status)}</p></article>`;
    }).join('');
  } catch (_) {
    $('live-signals').textContent = 'دریافت وضعیت سیگنال زنده ناموفق بود؛ وضعیت قبلی معتبر فرض نمی‌شود.';
  }
}

function controls(disabled) {
  $('refresh').disabled = disabled;
  $('run-select').disabled = disabled;
  document.querySelectorAll('[data-symbol], [data-asset]').forEach(button => button.disabled = disabled);
}

function showError(error) {
  $('error').textContent = `${error.message}؛ دوباره «به‌روزرسانی» را بزنید. داده‌های نمایان ممکن است قدیمی باشند.`;
  $('error').hidden = false;
  $('status').textContent = 'دریافت نتایج ناموفق بود.';
}

async function selectSymbol(symbol) {
  if (state.busy) return;
  const previous = state.symbol;
  state.symbol = symbol;
  const run = state.runs.find(item => item.symbol === symbol) || null;
  await selectRun(run, previous);
}

async function selectRun(run, previousSymbol = state.symbol) {
  if (state.busy) return;
  state.busy = true;
  controls(true);
  try {
    const trades = run && run.trade_count > 0 ? await get(`/trades/${run.run_id}`) : [];
    state.run = run;
    state.trades = trades;
    state.page = 0;
    $('trade-filter').value = 'all';
    $('error').hidden = true;
    render();
  } catch (error) {
    state.symbol = previousSymbol;
    $('run-select').value = String(state.run?.run_id || '');
    showError(error);
  } finally {state.busy = false; controls(false);}
}

function render() {
  assets();
  document.querySelectorAll('[data-symbol]').forEach(button => {
    const selected = button.dataset.symbol === state.symbol;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', selected);
  });
  $('run-select').innerHTML = state.runs.filter(run => run.symbol === state.symbol).map(run => `<option value="${run.run_id}" ${state.run?.run_id === run.run_id ? 'selected' : ''}>اجرای #${run.run_id} · ${run.train_rows} نمونهٔ آموزش</option>`).join('');
  $('results').hidden = !state.run;
  $('empty').hidden = !!state.run;
  if (!state.run) return;
  const run = state.run;
  const metrics = [
    ['بازده خالص مرکب', signed(run.total_return_compounded), 'پس از کارمزد و لغزش', tone(run.total_return_compounded)],
    ['نرخ برد', percent(run.win_rate), `${state.trades.filter(trade => trade.net_return > 0).length} معاملهٔ سودده از ${run.trade_count}`, ''],
    ['بیشترین افت سرمایه', percent(run.max_drawdown), 'نسبت به اوج قبلی سرمایه', 'negative'],
    ['تعداد معاملات', number(run.trade_count), 'معاملات بدون هم‌پوشانی', '']
  ];
  $('metrics').innerHTML = metrics.map(([label, value, detail, color]) => `<article class="metric"><div class="metric-label">${label}</div><div class="metric-number ${color}">${value}</div><div class="metric-detail">${detail}</div></article>`).join('');
  $('quality').innerHTML = `<div class="quality-main"><div>دقت طبقه‌بندی<small>آستانهٔ طبقه‌بندی: ۰٫۵</small></div><strong>${percent(run.test_accuracy)}</strong></div><div class="quality-bar"><i style="width:${run.test_accuracy * 100}%"></i></div><div class="quality-row"><span>AUC · تفکیک دو کلاس</span><strong>${run.test_auc == null ? '—' : run.test_auc.toFixed(3)}</strong></div><div class="quality-row"><span>Brier · کمتر بهتر است</span><strong>${run.test_brier_score.toFixed(3)}</strong></div><div class="quality-row"><span>ضریب سود</span><strong>${run.profit_factor == null ? '—' : run.profit_factor.toFixed(2)}</strong></div><div class="split-bar">${[run.train_rows,run.validation_rows,run.test_rows].map(size=>`<i style="flex:${size}"></i>`).join('')}</div><div class="split-label"><span>آموزش ${number(run.train_rows)}</span><span>اعتبارسنجی ${number(run.validation_rows)}</span><span>آزمون ${number(run.test_rows)}</span></div>`;
  let equity = 100, peak = 100;
  const capital = [100], drawdown = [0];
  state.trades.forEach(trade => {equity *= 1 + trade.net_return; peak = Math.max(peak, equity); capital.push(equity); drawdown.push((equity / peak - 1) * 100);});
  lineChart('equity-chart', capital, '#f4b45e', false);
  lineChart('drawdown-chart', drawdown, '#fc7c8e', true);
  barChart(state.trades.map(trade => trade.net_return * 100));
  $('trade-total').textContent = `${run.trade_count} معامله`;
  $('method-details').textContent = `کندل ۱۵ دقیقه · افق ${run.horizon_bars} کندل (${run.horizon_bars * 15 / 60} ساعت) · آستانهٔ سیگنال ${run.signal_probability_threshold} · کارمزد هر سمت ${run.fees_bps_per_side / 100}% · لغزش هر سمت ${run.slippage_bps_per_side / 100}% · تقسیم زمانی آموزش / اعتبارسنجی / آزمون با فاصلهٔ پاک‌سازی.`;
  table();
}

function chartFrame(id, values, color, suffix) {
  const width = 620, height = id === 'equity-chart' ? 220 : 170;
  const left = 54, right = 20, top = 14, bottom = 27;
  let min = Math.min(...values), max = Math.max(...values);
  if (id === 'returns-chart') {min = Math.min(min,0); max = Math.max(max,0);}
  const padding = Math.max((max - min) * .18, suffix ? .05 : .1);
  min -= padding; max += padding;
  const x = index => left + index * (width - left - right) / Math.max(values.length - 1, 1);
  const y = value => top + (max - value) * (height - top - bottom) / (max - min);
  let markup = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${safe($(id).getAttribute('aria-label'))}"><defs><linearGradient id="gradient-${id}" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="${color}" stop-opacity=".2"/><stop offset="100%" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>`;
  for (let i = 0; i < 5; i++) {
    const value = min + (max - min) * i / 4;
    markup += `<line x1="${left}" x2="${width-right}" y1="${y(value)}" y2="${y(value)}" stroke="#263244" stroke-dasharray="3 5"/><text x="${left-8}" y="${y(value)+4}" text-anchor="end">${value.toFixed(2)}${suffix}</text>`;
  }
  const labels = [...new Set([0,Math.floor((values.length - 1)/2),values.length - 1])];
  labels.forEach(index => markup += `<text x="${x(index)}" y="${height-5}" text-anchor="middle">${index === 0 && id !== 'returns-chart' ? 'START' : `#${index + (id === 'returns-chart' ? 1 : 0)}`}</text>`);
  return {markup,width,height,left,right,top,bottom,x,y};
}

function lineChart(id, values, color, isPercent) {
  if (state.trades.length === 0) {$(id).innerHTML = '<div class="chart-empty">با این آستانه معامله‌ای ثبت نشده است.</div>'; return;}
  const frame = chartFrame(id,values,color,isPercent ? '%' : '');
  const path = values.map((value,index)=>`${index === 0 ? 'M' : 'L'} ${frame.x(index)} ${frame.y(value)}`).join(' ');
  const baseline = isPercent ? frame.y(0) : frame.height - frame.bottom;
  let markup = frame.markup + `<path d="${path} L ${frame.x(values.length-1)} ${baseline} L ${frame.x(0)} ${baseline} Z" fill="url(#gradient-${id})"/><path d="${path}" stroke="${color}" stroke-width="2.4" fill="none" stroke-linejoin="round"/>`;
  values.forEach((value,index)=> markup += `<circle class="chart-point" cx="${frame.x(index)}" cy="${frame.y(value)}" r="3" fill="${color}" tabindex="0"><title>${index === 0 ? 'شروع' : `معامله ${index}`} · ${value.toFixed(2)}${isPercent ? '%' : ''}${index ? ` · ${date(state.trades[index-1].exit_time)}` : ''}</title></circle>`);
  $(id).innerHTML = markup + '</svg>';
}

function barChart(values) {
  if (!values.length) {$('returns-chart').innerHTML = '<div class="chart-empty">معامله‌ای ثبت نشده است.</div>'; return;}
  const frame = chartFrame('returns-chart',values,'#57d6ac','%');
  const step = (frame.width-frame.left-frame.right) / values.length;
  let markup = frame.markup + `<line x1="${frame.left}" x2="${frame.width-frame.right}" y1="${frame.y(0)}" y2="${frame.y(0)}" stroke="#8190a6"/>`;
  values.forEach((value,index)=>markup += `<rect x="${frame.left+index*step+step*.16}" y="${Math.min(frame.y(value),frame.y(0))}" width="${step*.68}" height="${Math.max(Math.abs(frame.y(value)-frame.y(0)),1)}" rx="2" fill="${value>0 ? '#57d6ac' : '#fc7c8e'}" tabindex="0"><title>معامله ${index+1} · ${value.toFixed(2)}% · ${date(state.trades[index].exit_time)}</title></rect>`);
  $('returns-chart').innerHTML = markup + '</svg>';
}

function filteredTrades() {
  const filter = $('trade-filter').value;
  return state.trades.map((trade,index)=>({...trade,index:index+1})).filter(trade => filter === 'all' || (filter === 'win' && trade.net_return>0) || (filter === 'loss' && trade.net_return<0) || trade.result === filter);
}

function table() {
  const rows = filteredTrades();
  const pages = Math.max(1,Math.ceil(rows.length/10));
  state.page = Math.min(state.page,pages-1);
  const reasons = {TP:'حد سود',SL:'حد ضرر',TIME:'پایان زمان'};
  $('trade-rows').innerHTML = rows.slice(state.page*10,state.page*10+10).map(trade=>`<tr><td class="numeric">${trade.index}</td><td>${date(trade.entry_time)}</td><td>${date(trade.exit_time)}</td><td class="numeric">${number(trade.entry_price)}</td><td class="numeric">${number(trade.exit_price)}</td><td class="numeric">${trade.probability.toFixed(3)}</td><td><span class="exit-tag ${safe(trade.result)}">${safe(reasons[trade.result] || trade.result)}</span></td><td class="numeric ${tone(trade.net_return)}">${signed(trade.net_return)}</td></tr>`).join('') || '<tr><td colspan="8" style="text-align:center;color:#8b99ae">معامله‌ای با این فیلتر وجود ندارد.</td></tr>';
  $('table-count').textContent = `${rows.length} معامله · اجرای #${state.run.run_id}`;
  $('page-number').textContent = `${state.page+1} / ${pages}`;
  $('previous').disabled = state.page === 0;
  $('next').disabled = state.page >= pages-1;
  $('export').disabled = rows.length === 0;
}

$('refresh').addEventListener('click',refresh);
document.querySelectorAll('[data-symbol]').forEach(button=>button.addEventListener('click',()=>selectSymbol(button.dataset.symbol)));
$('run-select').addEventListener('change',()=>selectRun(state.runs.find(run=>run.run_id === Number($('run-select').value))));
$('trade-filter').addEventListener('change',()=>{state.page=0;table();});
$('previous').addEventListener('click',()=>{state.page--;table();});
$('next').addEventListener('click',()=>{state.page++;table();});
$('export').addEventListener('click',()=>{
  const keys = ['symbol','entry_time','exit_time','entry_price','exit_price','probability','result','net_return'];
  const rows = filteredTrades().map(trade=>keys.map(key=>key === 'symbol' ? state.symbol : key.endsWith('_time') ? new Date(trade[key]).toISOString() : trade[key]).join(','));
  const blob = new Blob(['\uFEFF'+keys.join(',')+'\r\n'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href=url;link.download=`${state.symbol}-run-${state.run.run_id}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
refresh();
setInterval(()=>{if (!document.hidden) refresh();},60000);
