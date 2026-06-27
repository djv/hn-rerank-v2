// Dashboard JS extracted from templates/index.html (no build step, vanilla JS).
// Restore temporary client feedback state before regeneration updates the static file.
// Also clean up keys that have been persisted on the server or relate to discarded cards.
const USER_TOKEN = document.cookie.match(/hn_token=([^;]+)/)?.[1] || '';
const fbKey = (sid) => USER_TOKEN + '_fb_' + sid;

const presentStoryIds = new Set();
document.querySelectorAll('.story-card').forEach(card => {
  const storyId = card.dataset.storyId;
  presentStoryIds.add(storyId);

  const storedAction = localStorage.getItem(fbKey(storyId));
  if (storedAction) {
    const htmlVotedAction = card.dataset.voted;

    if (htmlVotedAction === storedAction) {
      localStorage.removeItem(fbKey(storyId));
    } else {
      card.dataset.voted = storedAction;
    }
  }
});

// Garbage collect stale localStorage entries for stories no longer on the page
for (let i = 0; i < localStorage.length; i++) {
  const key = localStorage.key(i);
  if (key && key.startsWith(USER_TOKEN + '_fb_')) {
    const storyId = key.replace(USER_TOKEN + '_fb_', '');
    if (!presentStoryIds.has(storyId)) {
      localStorage.removeItem(key);
      i--;
    }
  }
}

// Swipe deck state
const storiesContainer = document.getElementById('stories');
const refreshBanner = document.getElementById('refresh-banner');
const refreshBannerText = document.getElementById('refresh-banner-text');
const refreshNowBtn = document.getElementById('refresh-now-btn');
const modeTabs = Array.from(document.querySelectorAll('[data-mode]'));
const sourceTabs = Array.from(document.querySelectorAll('[data-source]'));
const voteBar = document.querySelector('.vote-bar');
let pendingFeedbackRequests = 0;
let refillQueued = false;
let isRefilling = false;
let activeCard = null;
let lastVote = null;
let currentMode = 'default';
let currentSource = 'mixed';
let votesSinceRankingRefresh = 0;
let preloadedRefillDoc = null;
let isPreloadingRefill = false;
let preloadRefillPromise = null;
const MAX_QUEUE = 12;
const LOW_WATERMARK = 4;
const PREFETCH_COUNT = 3;
const IDLE_INACTIVE_PREFETCH_COUNT = 3;
const VOTES_PER_RANKING_REFRESH = 5;
let idleModePrefetchHandle = null;

function cards() {
  return Array.from(storiesContainer.querySelectorAll('.story-card'));
}

function matchesCurrentSource(card) {
  if (currentSource === 'hn') {
    return card.dataset.isHn === '1';
  }
  if (currentSource === 'non-hn') {
    return card.dataset.isHn === '0';
  }
  return true;
}

function matchesCurrentMode(card) {
  if (!matchesCurrentSource(card)) return false;
  if (currentMode === 'popular') {
    return card.dataset.modePopular === '1';
  }
  if (currentMode === 'explore') {
    return card.dataset.modeExplore === '1';
  }
  return true;
}

function queuedCards() {
  return cards().filter(card => !card.dataset.voted && matchesCurrentMode(card));
}

// Map a story's rank position (sorted by data-score desc) to a hue.
// Rank 1 (top) = blue, rank N (bottom) = red. Rank-percentile mapping
// (rather than linear-in-score) ensures visually distinguishable colors
// even when the score distribution clusters near 1.0.
function applyGradient() {
  const cards = Array.from(document.querySelectorAll('.story-card'));
  const N = cards.length;
  if (N <= 1) return;
  const ranked = cards
    .map(card => ({ card, score: parseFloat(card.dataset.score) || 0 }))
    .sort((a, b) => b.score - a.score);
  ranked.forEach((entry, i) => {
    // i=0 -> top rank -> blue (220); i=N-1 -> bottom -> red (0)
    const hue = 220 * (N - 1 - i) / (N - 1);
    entry.card.style.borderLeftColor = `hsl(${hue.toFixed(1)}, 70%, 40%)`;
    entry.card.style.backgroundColor = `color-mix(in srgb, var(--pico-card-background-color) 96%, hsl(${hue.toFixed(1)}, 70%, 50%) 4%)`;
  });
}

function orderByRank() {
  const cards = Array.from(document.querySelectorAll('.story-card'));
  cards.sort((a, b) => parseFloat(b.dataset.score) - parseFloat(a.dataset.score));
  cards.forEach(card => storiesContainer.appendChild(card));
}

function orderByDate() {
  const cards = Array.from(document.querySelectorAll('.story-card'));
  cards.sort((a, b) => Number(b.dataset.time || 0) - Number(a.dataset.time || 0));
  cards.forEach(card => storiesContainer.appendChild(card));
}

function shuffleStories() {
  const cards = Array.from(document.querySelectorAll('.story-card'));
  for (let i = cards.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [cards[i], cards[j]] = [cards[j], cards[i]];
  }
  cards.forEach(card => storiesContainer.appendChild(card));
}

function orderForCurrentMode() {
  if (currentMode === 'date') orderByDate();
  else if (currentMode === 'default') orderByRank();
  else shuffleStories();
}

function updateRefreshProgress() {
  const filledCount = votesSinceRankingRefresh;
  const segments = document.querySelectorAll('.vote-bar .refresh-segment');
  segments.forEach((seg, idx) => {
    const isFilled = idx < filledCount;
    seg.classList.toggle('filled', isFilled);
    seg.classList.toggle('next-refresh', filledCount === VOTES_PER_RANKING_REFRESH && isFilled);
  });
  const progressBar = document.querySelector('.vote-bar .refresh-progress');
  progressBar?.setAttribute('aria-valuenow', String(Math.min(filledCount, VOTES_PER_RANKING_REFRESH)));
}

function updateQueueStatus() {
  updateRefreshProgress();
}

function setActiveCard(card) {
  cards().forEach(c => c.classList.toggle('active', c === card));
  activeCard = card || null;
  if (activeCard) {
    openTldrDetail(activeCard);
    voteBar.hidden = false;
    updateVoteBar();
  } else {
    document.querySelectorAll('[data-fb]').forEach(btn => btn.classList.remove('active'));
    voteBar.hidden = true;
  }
  prefetchUpcomingTldrs();
  scheduleIdleModePrefetch();
  updateQueueStatus();
  maybeRefillQueue();
}

function updateVoteBar() {
  document.querySelectorAll('[data-fb]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.fb === activeCard?.dataset.voted);
  });
}

function showNextCard() {
  setActiveCard(queuedCards()[0] || null);
}

function setMode(mode) {
  currentMode = mode;
  modeTabs.forEach(tab => {
    tab.classList.toggle('active', tab.dataset.mode === currentMode);
  });
  if (activeCard && !matchesCurrentMode(activeCard)) {
    activeCard.classList.remove('active');
  }
  if (refillQueued) {
    refillWhenReady();
  }
  orderForCurrentMode();
  showNextCard();
}

function setSource(source) {
  currentSource = source;
  sourceTabs.forEach(tab => {
    tab.classList.toggle('active', tab.dataset.source === currentSource);
  });
  if (activeCard && !matchesCurrentMode(activeCard)) {
    activeCard.classList.remove('active');
  }
  showNextCard();
  updateQueueStatus();
}

applyGradient();

// Snarkdown - 1KB markdown parser (https://github.com/developit/snarkdown)
const TAGS={'':['<em>','</em>'],_:['<strong>','</strong>'],'*':['<strong>','</strong>'],'~':['<s>','</s>'],'\n':['<br />'],' ':['<br />'],'-':['<hr />']};
function outdent(str){return str.replace(RegExp('^'+(str.match(/^(\t| )+/)||'')[0],'gm'),'')}
function encodeAttr(str){return(str+'').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function snarkdown(md,prevLinks){let tokenizer=/((?:^|\n+)(?:\n---+|\* \*(?: \*)+)\n)|(?:^``` *(\w*)\n([\s\S]*?)\n```$)|((?:(?:^|\n+)(?:\t|  {2,}).+)+\n*)|((?:(?:^|\n)([>*+-]|\d+\.)\s+.*)+)|(?:!\[([^\]]*?)\]\(([^)]+?)\))|(\[)|(\](?:\(([^)]+?)\))?)|(?:(?:^|\n+)([^\s].*)\n(-{3,}|={3,})(?:\n+|$))|(?:(?:^|\n+)(#{1,6})\s*(.+)(?:\n+|$))|(?:`([^`].*?)`)|(  \n\n*|\n{2,}|__|\*\*|[_*]|~~)/gm,context=[],out='',links=prevLinks||{},last=0,chunk,prev,token,inner,t;function tag(token){let desc=TAGS[token[1]||''],end=context[context.length-1]==token;if(!desc)return token;if(!desc[1])return desc[0];if(end)context.pop();else context.push(token);return desc[end|0]}function flush(){let str='';while(context.length)str+=tag(context[context.length-1]);return str}md=md.replace(/^\[(.+?)\]:\s*(.+)$/gm,(s,name,url)=>{links[name.toLowerCase()]=url;return''}).replace(/^\n+|\n+$/g,'');while((token=tokenizer.exec(md))){prev=md.substring(last,token.index);last=tokenizer.lastIndex;chunk=token[0];if(prev.match(/[^\\](\\\\)*\\$/)){}else if(t=(token[3]||token[4])){chunk='<pre class="code '+(token[4]?'poetry':token[2].toLowerCase())+'"><code'+(token[2]?` class="language-${token[2].toLowerCase()}"`:'')+'>'+outdent(encodeAttr(t).replace(/^\n+|\n+$/g,''))+'</code></pre>'}else if(t=token[6]){if(t.match(/\./)){token[5]=token[5].replace(/^\d+/gm,'')}inner=snarkdown(outdent(token[5].replace(/^\s*[>*+.-]/gm,'')));if(t=='>')t='blockquote';else{t=t.match(/\./)?'ol':'ul';inner=inner.replace(/^(.*)(\n|$)/gm,'<li>$1</li>')}chunk='<'+t+'>'+inner+'</'+t+'>'}else if(token[8]){chunk=`<img src="${encodeAttr(token[8])}" alt="${encodeAttr(token[7])}">`}else if(token[10]){out=out.replace('<a>',`<a href="${encodeAttr(token[11]||links[prev.toLowerCase()])}">`);chunk=flush()+'</a>'}else if(token[9]){chunk='<a>'}else if(token[12]||token[14]){t='h'+(token[14]?token[14].length:(token[13]>'='?1:2));chunk='<'+t+'>'+snarkdown(token[12]||token[15],links)+'</'+t+'>'}else if(token[16]){chunk='<code>'+encodeAttr(token[16])+'</code>'}else if(token[17]||token[1]){chunk=tag(token[17]||'--')}out+=prev;out+=chunk}return(out+md.substring(last)+flush()).replace(/^\n+|\n+$/g,'')}

function normalizeTldrMarkdown(text) {
  return text
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map(line => {
      const trimmed = line.trim();
      const isPlainHeading =
        trimmed.length > 0 &&
        trimmed.length <= 48 &&
        !/^[#\-*>`]/.test(trimmed) &&
        !/[.:,]$/.test(trimmed) &&
        trimmed.split(/\s+/).length <= 5 &&
        /[A-Za-z]/.test(trimmed);
      if (isPlainHeading) {
        return `### ${trimmed}`;
      }
      const labelMatch = trimmed.match(/^([A-Z][A-Za-z ]{1,40}):\s*(.*)$/);
      if (labelMatch) {
        return `- **${labelMatch[1]}:** ${labelMatch[2]}`;
      }
      return line;
    })
    .join('\n')
    .replace(/(\S)\s+-\s+(?=\S)/g, '$1\n- ');
}

function parseSimpleMarkdown(text) {
  return snarkdown(normalizeTldrMarkdown(text));
}

function styleTldrLabels(container) {
  // Backup for the rare case the LLM still produces `- **Label**:` despite
  // the prompt. Adds `.tldr-label` so CSS can style it differently.
  container.querySelectorAll('li').forEach(li => {
    if (li.querySelector(':scope > ul, :scope > ol')) return;
    const text = li.cloneNode(true).textContent.trim();
    if (text.endsWith(':')) li.classList.add('tldr-label');
  });
}

function updateRefreshBanner() {
  const hasPending = pendingFeedbackRequests > 0;
  const isReady =
    refillQueued &&
    preloadedRefillDoc !== null &&
    !isRefilling &&
    !hasPending;
  const isPreparing =
    refillQueued &&
    !isReady &&
    !isRefilling &&
    !hasPending;
  const shouldShow = hasPending || isRefilling || isPreparing || isReady;

  refreshBanner.hidden = !shouldShow;
  refreshNowBtn.hidden = !isReady;

  if (!shouldShow) return;
  if (isRefilling) {
    refreshBannerText.textContent = 'Refilling queue...';
  } else if (hasPending) {
    const label = pendingFeedbackRequests === 1 ? '1 vote saving' : `${pendingFeedbackRequests} votes saving`;
    refreshBannerText.textContent = `${label} syncing`;
  } else if (isPreparing) {
    refreshBannerText.textContent = 'Preparing refresh...';
  } else if (isReady) {
    refreshBannerText.textContent = 'New ranking ready';
  }
}

function markVoteSaving() {
  updateRefreshBanner();
}

async function refillWhenReady() {
  if (isRefilling) return;
  refillQueued = true;
  updateRefreshBanner();
  if (pendingFeedbackRequests > 0) return;

  isRefilling = true;
  updateRefreshBanner();
  updateQueueStatus();
  try {
    await refillQueue();
  } catch (err) {
    console.error('Failed to refill queue', err);
  } finally {
    isRefilling = false;
    refillQueued = false;
    updateQueueStatus();
    updateRefreshBanner();
  }
}

refreshNowBtn.addEventListener('click', refillWhenReady);
modeTabs.forEach(tab => {
  tab.addEventListener('click', () => setMode(tab.dataset.mode));
});
sourceTabs.forEach(tab => {
  tab.addEventListener('click', () => setSource(tab.dataset.source));
});
document.querySelectorAll('[data-key-action]').forEach(btn => {
  const runKeyAction = () => {
    const action = btn.dataset.keyAction;
    if (action === 'undo') {
      undoLastVote();
    } else {
      submitVote(action);
    }
  };
  btn.addEventListener('click', runKeyAction);
  btn.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      runKeyAction();
    }
  });
});

async function fetchRefillDoc() {
  const resp = await fetch(window.location.href, { cache: 'no-store' });
  if (!resp.ok) {
    throw new Error(`Refill failed: ${resp.status}`);
  }
  const htmlText = await resp.text();
  const parser = new DOMParser();
  return parser.parseFromString(htmlText, 'text/html');
}

function scheduleRefillPreload() {
  if (preloadRefillPromise || preloadedRefillDoc) {
    return;
  }
  isPreloadingRefill = true;
  preloadRefillPromise = new Promise(resolve => {
    window.setTimeout(resolve, 2500);
  }).then(async () => {
    preloadedRefillDoc = await fetchRefillDoc();
    return preloadedRefillDoc;
  }).catch(err => {
    console.error('Failed to preload refreshed queue', err);
    return null;
  }).finally(() => {
    preloadRefillPromise = null;
    isPreloadingRefill = false;
    updateRefreshBanner();
  });
}

async function refillQueue({ forceFetch = false } = {}) {
  let newDoc = preloadedRefillDoc;
  if (forceFetch || !newDoc) {
    if (!forceFetch && preloadRefillPromise) {
      newDoc = await preloadRefillPromise;
    }
    if (!newDoc) {
      newDoc = await fetchRefillDoc();
    }
  }
  preloadedRefillDoc = null;
  const newStories = newDoc.getElementById('stories');
  if (!newStories) {
    throw new Error('Refill failed: stories container missing');
  }

  cards().forEach(card => {
    if (card !== activeCard) {
      card.remove();
    }
  });
  const existing = new Set(cards().map(card => card.dataset.storyId));
  const incoming = Array.from(newStories.querySelectorAll('.story-card'));
  for (const card of incoming) {
    const storyId = card.dataset.storyId;
    if (!storyId || existing.has(storyId) || localStorage.getItem(fbKey(storyId))) {
      continue;
    }
    storiesContainer.appendChild(card);
    existing.add(storyId);
  }

  applyGradient();
  bindEvents();
  if (!activeCard || activeCard.dataset.voted) {
    showNextCard();
  }
}

// Function to bind click event listeners to current cards
function bindEvents() {
  // 1. Feedback buttons
  document.querySelectorAll('[data-fb]').forEach(btn => {
    btn.onclick = () => {
      submitVote(btn.dataset.fb);
    };
  });

  // 2. TLDR detail buttons and empty card space
  document.querySelectorAll('.story-card').forEach(card => {
    card.onclick = async (e) => {
      if (
        e.target.closest('a, button, input, textarea, select, summary, [role="button"], .tldr-detail-content') ||
        window.getSelection()?.toString()
      ) {
        return;
      }
      await openTldrDetail(card);
    };
  });

  document.querySelectorAll('[data-tldr-detail]').forEach(btn => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      await openTldrDetail(btn.closest('.story-card'));
    };
  });
}



function maybeRefillQueue() {
  if (queuedCards().length > LOW_WATERMARK || isRefilling || refillQueued) {
    return;
  }
  if (!activeCard) {
    refillWhenReady();
  }
}

function prefetchUpcomingTldrs() {
  queuedCards().slice(0, PREFETCH_COUNT).forEach(card => {
    if (!card.querySelector('.tldr-detail-content')) {
      openTldrDetail(card);
    }
  });
}

function cardsForMode(mode) {
  const filtered = cards().filter(card => {
    if (card.dataset.voted) return false;
    if (mode === 'popular') return card.dataset.modePopular === '1';
    if (mode === 'explore') return card.dataset.modeExplore === '1';
    return true;
  });
  if (mode === 'date') {
    filtered.sort((a, b) => Number(b.dataset.time || 0) - Number(a.dataset.time || 0));
  }
  return filtered;
}

function scheduleIdleModePrefetch() {
  if (idleModePrefetchHandle) {
    return;
  }
  const run = () => {
    idleModePrefetchHandle = null;
    ['default', 'popular', 'explore', 'date']
      .filter(mode => mode !== currentMode)
      .forEach(mode => {
        cardsForMode(mode)
          .slice(0, IDLE_INACTIVE_PREFETCH_COUNT)
          .forEach(card => {
            if (!card.querySelector('.tldr-detail-content')) {
              openTldrDetail(card);
            }
          });
      });
  };
  if ('requestIdleCallback' in window) {
    idleModePrefetchHandle = window.requestIdleCallback(run, { timeout: 2500 });
  } else {
    idleModePrefetchHandle = window.setTimeout(run, 1500);
  }
}

function sendFeedback(storyId, action, queueRemaining, refreshRanking = false) {
  const apiPath = window.location.pathname.startsWith('/rewrite') ? '/rewrite/api/feedback' : '/api/feedback';
  return fetch(apiPath, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      story_id: storyId,
      action,
      queue_remaining: queueRemaining,
      refresh_ranking: refreshRanking
    })
  }).then(resp => {
    if (!resp.ok) {
      throw new Error(`Feedback failed: ${resp.status}`);
    }
    return resp.json();
  });
}

function submitVote(action, card = activeCard) {
  if (!card || card.dataset.voted) {
    return;
  }

  const storyId = Number(card.dataset.storyId);
  const applyExistingRefillAfterVote = refillQueued && pendingFeedbackRequests === 0;
  votesSinceRankingRefresh += 1;
  updateRefreshProgress();
  const shouldRefreshRanking = votesSinceRankingRefresh >= VOTES_PER_RANKING_REFRESH;
  if (shouldRefreshRanking) {
    votesSinceRankingRefresh = 0;
    updateRefreshProgress();
  }
  card.dataset.voted = action;
  localStorage.setItem(fbKey(storyId), action);
  document.querySelectorAll('[data-fb]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.fb === action);
  });
  const countEl = document.querySelector(`[data-vote-count="${action}"]`);
  if (countEl) {
    countEl.textContent = String(Number(countEl.textContent || '0') + 1);
  }

  if (action === 'up') {
    card.style.setProperty('--swipe-exit-x', '42px');
    card.style.setProperty('--swipe-exit-rotate', '1.5deg');
  } else if (action === 'down') {
    card.style.setProperty('--swipe-exit-x', '-42px');
    card.style.setProperty('--swipe-exit-rotate', '-1.5deg');
  } else {
    card.style.setProperty('--swipe-exit-x', '0');
    card.style.setProperty('--swipe-exit-rotate', '0deg');
  }
  card.classList.add('fade-out');

  window.setTimeout(() => {
    if (card.dataset.voted === action) {
      card.remove();
      showNextCard();
    }
  }, 150);

  const queueRemaining = queuedCards().length;
  pendingFeedbackRequests += 1;
  markVoteSaving();

  const savePromise = sendFeedback(
    storyId,
    action,
    queueRemaining,
    shouldRefreshRanking
  ).then(data => {
    if (data.ranking_refresh_queued) {
      refillQueued = true;
      scheduleRefillPreload();
    }
    return data;
  }).catch(err => {
    console.error('Network error submitting feedback', err);
    refreshBannerText.textContent = 'Vote failed to save';
    refreshBanner.hidden = false;
    refreshNowBtn.hidden = true;
  }).finally(() => {
    pendingFeedbackRequests = Math.max(0, pendingFeedbackRequests - 1);
    if (applyExistingRefillAfterVote && pendingFeedbackRequests === 0) {
      window.setTimeout(refillWhenReady, 250);
    } else {
      updateRefreshBanner();
      maybeRefillQueue();
    }
  });

  lastVote = { storyId, action, card, savePromise };
}

function undoLastVote() {
  const vote = lastVote;
  if (!vote) {
    return;
  }
  lastVote = null;

  const { storyId, action, card, savePromise } = vote;
  delete card.dataset.voted;
  card.classList.remove('fade-out');
  localStorage.removeItem(fbKey(storyId));
  document.querySelectorAll('[data-fb]').forEach(btn => {
    btn.classList.remove('active');
  });
  if (!card.isConnected) {
    storiesContainer.insertBefore(card, storiesContainer.firstChild);
  }
  bindEvents();
  setActiveCard(card);

  pendingFeedbackRequests += 1;
  markVoteSaving();
  Promise.resolve(savePromise).finally(() => {
    return sendFeedback(storyId, 'clear', queuedCards().length, true);
  }).then(data => {
    if (data.ranking_refresh_queued) {
      refillQueued = true;
      preloadedRefillDoc = null;
      scheduleRefillPreload();
    }
  }).catch(err => {
    console.error('Network error undoing feedback', err);
    refreshBannerText.textContent = 'Undo failed to save';
    refreshBanner.hidden = false;
    refreshNowBtn.hidden = true;
  }).finally(() => {
    pendingFeedbackRequests = Math.max(0, pendingFeedbackRequests - 1);
    updateRefreshBanner();
  });
}

function openStoryUrl(kind) {
  const card = document.querySelector('.story-card.active, .story-card[data-active]')
            || document.querySelector('.story-card');
  if (!card) return;
  const attr = kind === 'article' ? 'articleUrl' : 'commentsUrl';
  const url = card.dataset[attr];
  if (!url) return;
  window.open(url, '_blank', 'noopener,noreferrer');
}

async function openTldrDetail(card) {
  if (!card) {
    return;
  }

  const btn = card.querySelector('[data-tldr-detail]');
  const storyId = Number(card.dataset.storyId);
  const storyTitle = card.querySelector('.story-title a')?.textContent || '';

  let contentDiv = card.querySelector('.tldr-detail-content');
  if (contentDiv?.dataset.loading === 'true') {
    return;
  }
  if (contentDiv && contentDiv.style.display !== 'none') {
    return;
  }
  if (!contentDiv) {
    contentDiv = document.createElement('div');
    contentDiv.className = 'tldr-detail-content';
    contentDiv.style.marginTop = '0.4rem';
    contentDiv.style.fontSize = '0.85rem';
    contentDiv.style.borderTop = '1px dashed var(--pico-muted-border-color)';
    contentDiv.style.paddingTop = '0.4rem';
    contentDiv.style.display = 'none';
    card.appendChild(contentDiv);
  }
  contentDiv.dataset.loading = 'true';

  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
  }

  const apiPath = window.location.pathname.startsWith('/rewrite') ? '/rewrite/api/tldr-detail' : '/api/tldr-detail';
  try {
    const resp = await fetch(apiPath, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ story_id: storyId, story_title: storyTitle })
    });
    if (resp.ok) {
      const data = await resp.json();
      contentDiv.innerHTML = parseSimpleMarkdown(data.tldr);
      styleTldrLabels(contentDiv);
      contentDiv.style.display = 'block';
      if (btn) {
        btn.style.display = 'none';
      }
    } else {
      const errData = await resp.json().catch(() => ({}));
      if (btn) {
        btn.textContent = 'Failed';
        btn.title = errData.error || 'Failed to generate TLDR';
        btn.disabled = false;
      }
    }
  } catch (err) {
    console.error(err);
    if (btn) {
      btn.textContent = 'Error';
      btn.disabled = false;
    }
  } finally {
    contentDiv.dataset.loading = 'false';
  }
}

document.addEventListener('keydown', (e) => {
  if (
    e.repeat ||
    e.ctrlKey || e.metaKey || e.altKey ||
    e.target.closest?.('input, textarea, select, [contenteditable="true"]')
  ) {
    return;
  }
  const key = e.key.toLowerCase();
  if (key === 'arrowup' || key === 'arrowdown') {
    e.preventDefault();
    const card = document.querySelector('.story-card.active');
    if (card) {
      const scrollAmount = card.clientHeight * 0.8;
      card.scrollBy({ top: key === 'arrowdown' ? scrollAmount : -scrollAmount });
    }
  } else if (key === 'k') {
    e.preventDefault();
    submitVote('up');
  } else if (key === 'j') {
    e.preventDefault();
    submitVote('down');
  } else if (key === 'l') {
    e.preventDefault();
    submitVote('neutral');
  } else if (key === 'o') {
    e.preventDefault();
    openStoryUrl('article');
  } else if (key === 'c') {
    e.preventDefault();
    openStoryUrl('comments');
  } else if (key === 'u') {
    e.preventDefault();
    undoLastVote();
  }
});

// First-time tip dismissal
(function() {
  const tip = document.getElementById('first-time-tip');
  const flag = 'hn_first_tip_seen_v2';
  if (tip && !localStorage.getItem(flag)) {
    tip.hidden = false;
    tip.querySelector('[data-dismiss-tip]')?.focus();
    const dismiss = () => {
      tip.hidden = true;
      localStorage.setItem(flag, '1');
    };
    tip.querySelector('[data-dismiss-tip]')?.addEventListener('click', dismiss);
    tip.addEventListener('click', (e) => { if (e.target === tip) dismiss(); });
    tip.addEventListener('keydown', (e) => { if (e.key === 'Escape') dismiss(); });
  }
})();

// Run initial event binding
bindEvents();
setMode('default');