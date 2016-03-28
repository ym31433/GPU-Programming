#pragma once
void CountPosition(const char *text, int *pos, int text_size);
//void CountPosition(const char *text, int *pos, int* debug_tree, int text_size);
int ExtractHead(const int *pos, int *head, int text_size);
void Part3(char *text, int *pos, int *head, int text_size, int n_head);
