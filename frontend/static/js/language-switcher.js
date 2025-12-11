// Language switcher functionality
document.addEventListener('DOMContentLoaded', function() {
  // Check URL for language parameter and update language if present
  const urlParams = new URLSearchParams(window.location.search);
  const langParam = urlParams.get('lang');

  if (langParam && (langParam === 'en' || langParam === 'ur')) {
    // Store the selected language in localStorage
    localStorage.setItem('language', langParam);

    // Update the document language attribute
    document.documentElement.lang = langParam;

    // Dispatch a custom event to notify React components about the language change
    window.dispatchEvent(new CustomEvent('languageChanged', {
      detail: { lang: langParam }
    }));
  }

  // Add event listeners to language switcher items
  const languageItems = document.querySelectorAll('.language-switcher-item');

  languageItems.forEach(item => {
    item.addEventListener('click', function(e) {
      // Get the target language from the href
      const langMatch = this.getAttribute('href')?.match(/lang=(en|ur)/);
      if (langMatch) {
        const lang = langMatch[1];

        // Store the selected language in localStorage
        localStorage.setItem('language', lang);

        // Update the document language attribute
        document.documentElement.lang = lang;

        // Dispatch a custom event to notify React components about the language change
        window.dispatchEvent(new CustomEvent('languageChanged', {
          detail: { lang: lang }
        }));

        // Prevent default navigation and handle language change manually
        e.preventDefault();

        // Update the URL parameter without page reload
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('lang', lang);
        window.history.pushState({}, '', newUrl);

        // Reload the page to ensure all content is updated with the new language
        window.location.reload();
      }
    });
  });

  // Listen for language change events to update UI if needed
  window.addEventListener('languageChanged', function(e) {
    console.log('Language changed to:', e.detail.lang);
  });
});