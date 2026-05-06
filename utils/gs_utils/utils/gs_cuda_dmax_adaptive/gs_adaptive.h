void _gs_render(
    const float *sigmas,
    const float *coords,
    const float *colors,
    float *rendered_img,
    const int s, 
    const int h, 
    const int w, 
    const int c,
    const float *dmax // <--- Cambiato da float a float* (puntatore alla lista)
);

void _gs_render_backward(
    const float *sigmas,
    const float *coords,
    const float *colors,
    const float *grads,
    float *grads_sigmas,
    float *grads_coords,
    float *grads_colors,
    const int s, 
    const int h, 
    const int w, 
    const int c,
    const float *dmax // <--- Cambiato da float a float*
);