
(() => {
  const title = document.getElementById('title');
  const article = document.getElementById('text');
  const titleCount = document.getElementById('titleCount');
  const textCount = document.getElementById('textCount');
  const clearButton = document.getElementById('clearForm');
  const form = document.getElementById('newsForm');
  const mobileMenu = document.getElementById('mobileMenu');
  const navLinks = document.getElementById('navLinks');

  function updateCounts() {
    if (title && titleCount) {
      titleCount.textContent = title.value.length;
    }

    if (article && textCount) {
      textCount.textContent = article.value.length;
    }
  }

  title?.addEventListener('input', updateCounts);
  article?.addEventListener('input', updateCounts);

  clearButton?.addEventListener('click', () => {
    if (title) title.value = '';
    if (article) article.value = '';

    updateCounts();
    title?.focus();

    const results = document.getElementById('results');
    if (results) results.remove();
  });

  form?.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');

    if (button) {
      button.disabled = true;
      button.innerHTML = '<span>◌</span> Analyzing news…';
    }
  });

  mobileMenu?.addEventListener('click', () => {
    const open = navLinks?.classList.toggle('open') || false;
    mobileMenu.setAttribute('aria-expanded', String(open));
  });

  navLinks?.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
      mobileMenu?.setAttribute('aria-expanded', 'false');
    });
  });

  updateCounts();

  if (document.getElementById('results')) {
    setTimeout(() => {
      document.getElementById('results')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }, 150);
  }
})();
