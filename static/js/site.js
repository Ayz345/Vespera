(function () {
  // Mobile navigation toggle
  var toggle = document.querySelector('.nav-toggle');
  var nav = document.getElementById('site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', String(!open));
      nav.classList.toggle('is-open', !open);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && nav.classList.contains('is-open')) {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.focus();
      }
    });
  }

  // File inputs: show the chosen file name, support drag and drop
  document.querySelectorAll('.file-field').forEach(function (field) {
    var input = field.querySelector('input[type=file]');
    var label = field.querySelector('.file-field__name');
    if (!input || !label) return;
    var empty = label.textContent;

    function update() {
      var f = input.files && input.files[0];
      if (f) {
        var kb = f.size / 1024;
        label.textContent = f.name + ' · ' + (kb > 1024 ? (kb / 1024).toFixed(1) + ' MB' : Math.round(kb) + ' KB');
        field.classList.add('has-file');
      } else {
        label.textContent = empty;
        field.classList.remove('has-file');
      }
    }
    input.addEventListener('change', update);

    ['dragenter', 'dragover'].forEach(function (ev) {
      field.addEventListener(ev, function (e) { e.preventDefault(); field.classList.add('is-dragging'); });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      field.addEventListener(ev, function (e) { e.preventDefault(); field.classList.remove('is-dragging'); });
    });
    field.addEventListener('drop', function (e) {
      if (e.dataTransfer && e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        update();
      }
    });
    update();
  });

  // Long-running forms: lock the submit button and say what is happening
  document.querySelectorAll('form[data-busy-text]').forEach(function (form) {
    form.addEventListener('submit', function () {
      var btn = form.querySelector('button[type=submit]');
      if (!btn) return;
      btn.disabled = true;
      btn.dataset.idleText = btn.textContent;
      btn.textContent = form.dataset.busyText;
      form.classList.add('is-busy');
    });
  });
})();
