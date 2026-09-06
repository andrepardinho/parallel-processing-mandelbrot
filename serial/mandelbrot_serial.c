#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

/** 
 * Calcula o conjunto de Mandelbrot para uma região específica do plano complexo.
 * @param count Array para armazenar o número de iterações para cada pixel.
 * @param width Largura da imagem.
 * @param height Altura da imagem.
 * @param max_iter Número máximo de iterações.
 * @param re_min Limite mínimo da parte real do plano complexo.
 * @param re_max Limite máximo da parte real do plano complexo.
 * @param im_min Limite mínimo da parte imaginária do plano complexo.
 * @param im_max Limite máximo da parte imaginária do plano complexo.
 */
void calcular_mandelbrot(int *count, int width, int height, int max_iter,
    double re_min, double re_max, double im_min, double im_max) {

    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {

            // Mapeia o pixel (px, py) para o plano complexo.
            // Exemplo: se px = 2048 e width = 4096, (double)px / width = 0.5,
            // ou seja, o meio do intervalo horizontal. Com re_min = -2.0 e
            // re_max = 1.0: c_real = -2.0 + 0.5 * (1.0 - (-2.0)) = -0.5.
            // mesma logica para c_imag usando im_min/im_max e py/height.
            double c_real = re_min + ((double)px / width) * (re_max - re_min);
            double c_imag = im_min + ((double)py / height) * (im_max - im_min);

            double z_real = 0.0, z_imag = 0.0;
            int iter = 0;

            while (iter < max_iter) {
                // Condição de escape: |z|^2 > 4.0 (equivalente a |z| > 2,
                // evitando a raiz quadrada).
                double z_real_sq = z_real * z_real;
                double z_imag_sq = z_imag * z_imag;

                if (z_real_sq + z_imag_sq > 4.0) break;
                
                double novo_z_imag = 2.0 * z_real * z_imag + c_imag; // (2ab) + c_imag
                z_real = z_real_sq - z_imag_sq + c_real; // (a^2 - b^2) + c_real
                z_imag = novo_z_imag;
                iter++;
            }
            count[(size_t)py * width + px] = iter;
        }
    }

}


void salvar_binario(const char *arquivo, int32_t *count, int width, int height) {
    FILE *f_bin = fopen(arquivo, "wb");
    if (!f_bin) {
        fprintf(stderr, "Erro ao abrir '%s' para escrita.\n", arquivo);
        return;
    }
    size_t n = (size_t)width * height;
    if (fwrite(count, sizeof(int32_t), n, f_bin) != n) {
        fprintf(stderr, "Erro ao escrever dados binários em '%s'.\n", arquivo);
    }
    fclose(f_bin);
}


void salvar_imagem_ppm(const char *arquivo, int32_t *count, int width, int height, int max_iter) {
    FILE *f_ppm = fopen(arquivo, "wb");
    if (!f_ppm) {
        fprintf(stderr, "Erro ao abrir '%s' para escrita.\n", arquivo);
        return;
    }
    fprintf(f_ppm, "P6\n%d %d\n255\n", width, height);
    size_t n = (size_t)width * height;
    for (size_t i = 0; i < n; i++) {
        unsigned char color = (unsigned char)(255 * count[i] / max_iter);
        fputc(color, f_ppm); // Red
        fputc(color, f_ppm); // Green
        fputc(color, f_ppm); // Blue
    }
    fclose(f_ppm);
}

static double tempo_atual_segundos(void) {
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1e9;
}

int main(void) {
    int width = 4096, height = 4096, max_iter = 1000;
    double re_min = -2.0, re_max = 1.0, im_min = -1.5, im_max = 1.5;

    // Matriz para armazenar o número de iterações de cada pixel.
    int32_t *count = (int32_t *)malloc((size_t)width * height * sizeof(int32_t));
    if (count == NULL) {
        fprintf(stderr, "Erro ao alocar memória para a matriz de contagem.\n");
        return 1;
    }

    double t_calc_ini, t_calc_fim, t_io_ini, t_io_fim;

    // Tempo de cálculo (o que entra em Speedup/Eficiência).
    t_calc_ini = tempo_atual_segundos();
    
    calcular_mandelbrot(count, width, height, max_iter, re_min, re_max, im_min, im_max);
    
    t_calc_fim = tempo_atual_segundos();

    // Tempo de escrita em disco separadamente
    t_io_ini = tempo_atual_segundos();
    
    salvar_binario("mandelbrot.bin", count, width, height);
    salvar_imagem_ppm("mandelbrot.ppm", count, width, height, max_iter);
    
    t_io_fim = tempo_atual_segundos();

    double tempo_calc = t_calc_fim - t_calc_ini;
    double tempo_io = t_io_fim - t_io_ini;

    printf("Resolucao: %dx%d, MAX_ITER=%d\n", width, height, max_iter);
    printf("Tempo de calculo: %.6f s\n", tempo_calc);
    printf("Tempo de I/O:     %.6f s\n", tempo_io);
    printf("Tempo total:      %.6f s\n", tempo_calc + tempo_io);

    free(count);
    return 0;
}