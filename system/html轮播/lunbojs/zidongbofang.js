var carousel = $('#videoCarousel');
var videos = carousel.find('video');
var currentVideoIndex = 0;

function playNextVideo() {
    // 停
    videos.each(function () {
        this.pause();
    });

    // 找
    var nextVideo = $(videos[currentVideoIndex]);
    nextVideo[0].play();

    // 自动轮播
    nextVideo.on('ended', function () {
        currentVideoIndex = (currentVideoIndex + 1) % videos.length;
        carousel.carousel(currentVideoIndex);
        playNextVideo(); // 下一个
    });
}

// 第一个视频
playNextVideo();

// 主观停
videos.on('play', function () {
    var videoIndex = videos.index(this);
    carousel.carousel(videoIndex);
});


