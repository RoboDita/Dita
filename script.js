let currentIndex = 0;
const itemsPerSlide = 4;

function showSlide(index) {
    const totalItems = document.querySelectorAll('.carousel-item').length;
    const maxIndex = Math.ceil(totalItems / itemsPerSlide) - 1;

    if (index > maxIndex) {
        currentIndex = 0;
    } else if (index < 0) {
        currentIndex = maxIndex;
    } else {
        currentIndex = index;
    }

    const carousel = document.querySelector('.carousel');
    const itemWidth = document.querySelector('.carousel-item').offsetWidth + 20; // 20px is the margin-right
    const totalWidth = itemWidth * itemsPerSlide;

    carousel.style.transform = `translateX(-${currentIndex * totalWidth}px)`;
}

function nextSlide() {
    showSlide(currentIndex + 1);
}

function prevSlide() {
    showSlide(currentIndex - 1);
}

// Initialize the slider
showSlide(currentIndex);