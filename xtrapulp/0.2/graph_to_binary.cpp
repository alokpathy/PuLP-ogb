#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <torch/torch.h>
#include <torch/script.h>

int main(int argc, char **argv) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <graph.pt>" << std::endl;
        return 1;
    }

    char BIN_PATH[80];
    memset(BIN_PATH, 0, 80 * sizeof(char));
    strcat(BIN_PATH, argv[1]);
    strcat(BIN_PATH, ".bin");

    torch::jit::script::Module container = torch::jit::load(argv[1]);
    // torch::jit::script::Module container = torch::jit::load("../ogbn-products.pt");
    torch::Tensor graph = container.attr("graph").toTensor();
    int32_t *graph_data = graph.data<int>();
    uint32_t *data = (uint32_t *) malloc(graph.size(0) * sizeof(uint32_t));

    std::cout << graph.size(0) << std::endl;

    for (uint32_t i = 0; i < graph.size(0); i++) {
        data[i] = (uint32_t) graph_data[i];

        if (i % 1000 == 0) {
            std::cout << "vertex: " << i << " total: " << graph.size(0) << std::endl;
        }
    }

    std::cout << BIN_PATH << std::endl;
    FILE *file = fopen(BIN_PATH, "wb");
    fwrite(data, graph.size(0) * sizeof(uint32_t), 1, file);
    fclose(file);

    free(data);

}
