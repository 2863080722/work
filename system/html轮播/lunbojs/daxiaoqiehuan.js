$(document).ready(function(){
    $('#videoCarousel').on('slide.bs.carousel', function (e) {
        var index = $(e.relatedTarget).index();
        var descriptionText = $('.carousel-item').eq(index).find('.video-description').text();
        $('#videoDescription').text(descriptionText); // 设置描述文本

        // Reset all videos to normal size
        $('.carousel-video').css('transform', 'scale(1)');

        // Enlarge the video that will be active after the transition
        var $nextVideo = $(e.relatedTarget).find('.carousel-video');
        $nextVideo.css('transform', 'scale(1.2)');

        // Pause all videos except the next one
        $('.carousel-video').each(function() {
            if (!$(this).is($nextVideo[0])) {
                this.pause();
            } else {
                this.play();
            }
        });
    });

    // 初始设置，放大并播放第一个视频
    $('.carousel-item.active .carousel-video').css('transform', 'scale(1.2)')[0].play();
    $('#videoDescription').text($('.carousel-item.active .video-description').text()); // 设置初始描述文本
});