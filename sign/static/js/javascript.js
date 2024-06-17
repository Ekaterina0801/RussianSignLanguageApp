document.addEventListener("DOMContentLoaded", function() {
    // Код JavaScript, который зависит от загрузки DOM-дерева
    document.getElementById('navbarToggle').addEventListener('click', function() {
        document.querySelector('nav').classList.toggle('open');
    });
});