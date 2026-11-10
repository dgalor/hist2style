window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    if (!dropdown || !button) return;

    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (!container || !dropdown || !button) return;
    if (!container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key !== 'Escape') return;
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    if (!dropdown || !button) return;
    dropdown.classList.remove('show');
    button.classList.remove('active');
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Copied!';

            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Copied!';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (!scrollButton) return;
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// One part of the page per screen: each part fades in as it arrives, its video loads shortly before it is reached, the dot of the
// part being looked at lights up, and videos in the other parts pause.
function setupParts() {
    var parts = Array.prototype.slice.call(document.querySelectorAll('.page-part'));
    if (!parts.length) return;
    var footer = document.querySelector('.footer');  // the last part and the footer share the final screen
    window.partsReady = true;
    if (!document.documentElement.classList.contains('parts-js')) {  // set in <head>; absent without IntersectionObserver or if this ran after the 4 s failsafe
        parts.forEach(function(part) { part.classList.add('is-visible', 'is-revealed'); loadVideos(part); });
        if (footer) footer.classList.add('is-visible');
        return;
    }
    function fitLast() { document.documentElement.style.setProperty('--footer-h', (footer ? footer.offsetHeight : 0) + 'px'); }
    fitLast();
    window.addEventListener('resize', fitLast);

    var nav = document.createElement('nav');
    nav.className = 'part-dots';
    nav.setAttribute('aria-label', 'Page sections');
    var dots = parts.map(function(part) {
        var dot = document.createElement('a');
        dot.href = '#' + part.id;
        dot.title = part.getAttribute('data-label') || part.id;
        dot.setAttribute('aria-label', dot.title);
        nav.appendChild(dot);
        return dot;
    });
    document.body.appendChild(nav);

    function show(part) {
        if (part.classList.contains('is-visible')) return;
        part.classList.add('is-visible');
        if (part.classList.contains('abstract-section')) part.classList.add('is-revealed');
        if (part === parts[parts.length - 1] && footer) footer.classList.add('is-visible');
        setTimeout(function() { loadVideos(part); }, PART_FADE_MS);  // the player loads once its part has faded in: a loading player can stall the fade
    }
    function observe() {
        var arrive = new IntersectionObserver(function(entries) {  // a part fades in once it is well into view
            entries.forEach(function(e) { if (e.isIntersecting) show(e.target); });
        }, { rootMargin: '0px 0px -15% 0px' });
        var focus = new IntersectionObserver(function(entries) {  // the part across the middle of the screen is the one being looked at
            entries.forEach(function(e) {
                if (!e.isIntersecting) return;
                var i = parts.indexOf(e.target);
                show(e.target);
                dots.forEach(function(dot, j) { dot.classList.toggle('is-active', j === i); });
                parts.forEach(function(part, j) { if (j !== i) pauseVideos(part); });
            });
        }, { rootMargin: '-49% 0px -49% 0px' });
        parts.forEach(function(part) { arrive.observe(part); focus.observe(part); });
    }
    setupSettle(parts);  // also under reduced motion, like the native scroll snapping it replaces
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { observe(); return; }  // everything is shown at once by the CSS
    whenFirstPartReady(parts[0], function() { show(parts[0]); observe(); });  // the title (in its final font) and the demo video appear together
}

// A soft pull to the nearest screen once scrolling stops: CSS scroll-snap (proximity) in Chrome pulls from 1/3 of the screen either way and
// lands in ~150 ms inside the wheel motion (measured 11-09); this waits until the reader stops and glides a little slower, from a little less far.
var SETTLE = { ahead: 0.3, behind: 0.3, idleMs: 120, glideMs: [280, 480] };  // distances are fractions of the screen height

function setupSettle(parts) {
    var root = document.documentElement, timer = null, anim = null, touching = false, pressing = false, dir = 0, lastY = window.scrollY, moveStart = null;
    function stop() {
        if (anim) cancelAnimationFrame(anim);
        anim = null;
        root.style.scrollBehavior = '';
        lastY = window.scrollY;
        moveStart = null;
    }
    function later() { clearTimeout(timer); timer = setTimeout(settle, SETTLE.idleMs); }
    function glide(target) {  // ease-in-out (SETTLE.glideMs); stops as soon as anything else moves the page (e.g. a scrollbar drag)
        var y0 = window.scrollY, d = target - y0, T = Math.min(SETTLE.glideMs[1], SETTLE.glideMs[0] + Math.abs(d) * 0.7), t0 = null, set = y0;
        root.style.scrollBehavior = 'auto';  // each step must land at once (the page's smooth scrolling would fight it)
        function step(now) {
            if (Math.abs(window.scrollY - set) > 2) return stop();
            if (t0 === null) t0 = now;
            var k = Math.min(1, (now - t0) / T);
            set = y0 + d * (0.5 - 0.5 * Math.cos(Math.PI * k));
            window.scrollTo(0, set);
            if (k < 1) anim = requestAnimationFrame(step); else stop();
        }
        anim = requestAnimationFrame(step);
    }
    function settle() {
        var vh = window.innerHeight, y = window.scrollY, maxY = root.scrollHeight - vh;
        if (touching || pressing || anim || y < 0 || y > maxY) return;
        moveStart = null;
        var stops = [maxY], inside = false;
        parts.forEach(function(p) {
            var top = Math.round(p.getBoundingClientRect().top + y), end = top + p.offsetHeight - vh;
            stops.push(Math.min(top, maxY));
            if (end > top + 1) {  // a part taller than the screen (phones): anywhere inside it is a resting place, its bottom edge a stop
                stops.push(Math.min(end, maxY));
                if (y >= top - 1 && y <= end + 1) inside = true;
            }
        });
        if (inside) return;
        var ahead = null, behind = null;
        stops.forEach(function(s) {
            var d = s - y;
            if (Math.abs(d) <= 1) return;
            if (d * dir > 0) { if (Math.abs(d) <= SETTLE.ahead * vh && (ahead === null || Math.abs(d) < Math.abs(ahead))) ahead = d; }
            else if (Math.abs(d) <= SETTLE.behind * vh && (behind === null || Math.abs(d) < Math.abs(behind))) behind = d;
        });
        var d = ahead !== null ? ahead : behind;
        if (d !== null) glide(y + d);
    }
    window.addEventListener('scroll', function() {
        if (anim) return;
        var y = window.scrollY;
        if (moveStart === null) moveStart = lastY;  // first scroll since the page came to rest: where this movement began
        if (Math.abs(y - lastY) > 0.5) dir = y > lastY ? 1 : -1;
        lastY = y;
        later();
    }, { passive: true });
    ['wheel', 'keydown'].forEach(function(type) { window.addEventListener(type, stop, { passive: true }); });  // the reader takes over at once
    window.addEventListener('touchstart', function() { touching = true; stop(); }, { passive: true });
    ['touchend', 'touchcancel'].forEach(function(type) { window.addEventListener(type, function() { touching = false; later(); }, { passive: true }); });
    window.addEventListener('pointerdown', function(e) { if (e.pointerType !== 'touch') { pressing = true; stop(); } }, { passive: true });
    ['pointerup', 'pointercancel'].forEach(function(type) {  // a touch that turns into a scroll fires pointercancel while the finger is still down
        window.addEventListener(type, function(e) { if (e.pointerType !== 'touch') { pressing = false; later(); } }, { passive: true });
    });
}

function whenFirstPartReady(part, done) {  // calls done once the part's video thumbnail and the web font (Inter) have loaded, or after 1.5 s at most
    var box = part.querySelector('.publication-video');
    var m = box && /url\(["']?([^"')]+)["']?\)/.exec(box.style.backgroundImage);
    var called = false, waiting = 1;
    function once() { if (!called) { called = true; done(); } }
    function one() { if (--waiting <= 0) once(); }
    if (m) {
        waiting++;
        var img = new Image();
        img.onload = img.onerror = one;
        img.src = m[1];
    }
    if (document.fonts && document.fonts.ready) { waiting++; document.fonts.ready.then(one, one); }
    one();
    setTimeout(once, 1500);
}

function loadVideos(part) {
    part.querySelectorAll('.publication-video iframe[data-src]').forEach(function(iframe) {
        if (iframe.getAttribute('src')) return;
        iframe.addEventListener('load', function() {  // fade the player in over the thumbnail once it has drawn
            setTimeout(function() { iframe.classList.add('is-loaded'); }, PLAYER_FADE_DELAY);
        }, { once: true });
        iframe.src = iframe.getAttribute('data-src');
    });
}
var PLAYER_FADE_DELAY = 300;  // the player draws its own thumbnail ~0-300 ms around its load event (site_proto_check.py --player-timeline)
var PART_FADE_MS = 750;  // .page-part fade (0.7 s) + margin

function pauseVideos(part) {
    part.querySelectorAll('.publication-video iframe[src]').forEach(function(iframe) {
        iframe.contentWindow.postMessage(JSON.stringify({ event: 'command', func: 'pauseVideo', args: [] }), 'https://www.youtube.com');
    });
}

function setupYouTubeEmbed() {  // opened as a local file, YouTube refuses to embed: show the thumbnail links instead
    if (window.location.protocol !== 'file:') return;
    var hint = document.getElementById('video-local-hint');
    document.querySelectorAll('.publication-video iframe[data-src]').forEach(function(iframe) {
        var fallback = iframe.parentElement.querySelector('.video-fallback');
        iframe.remove();
        if (fallback) fallback.classList.remove('is-hidden');
    });
    if (hint) hint.classList.remove('is-hidden');
}

function fitTitleVideo() {  // the demo video takes what the title block leaves of the first screen (CSS .title-video uses --title-h)
    var cols = document.querySelector('#top .hero-body .columns');
    if (cols) document.documentElement.style.setProperty('--title-h', cols.offsetHeight + 'px');
}

document.addEventListener('DOMContentLoaded', function() {
    setupYouTubeEmbed();
    fitTitleVideo();
    window.addEventListener('resize', fitTitleVideo);
    if (document.fonts) document.fonts.ready.then(fitTitleVideo);  // web fonts can change the title's height
    setupParts();
    var carouselOptions = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    };

    if (typeof bulmaCarousel !== 'undefined' && document.querySelector('.carousel')) {
        bulmaCarousel.attach('.carousel', carouselOptions);
    }

    if (typeof bulmaSlider !== 'undefined' && document.querySelector('input[type="range"].slider')) {
        bulmaSlider.attach();
    }

    setupVideoCarouselAutoplay();
});
