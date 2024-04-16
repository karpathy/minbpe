#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <math.h>
#include <memory>
#include <time.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <thread>

struct LinkedList {
    int *data;
    int capacity;
    int size;
};

// Function to initialize a new linked list
struct LinkedList createLinkedList(int *arr, int arr_size) {
    struct LinkedList newList = {arr, arr_size, arr_size};
    return newList;
}

// Function to remove a node from the list
void deleteElement(struct LinkedList *list, int index) {
    if (list->data[index] != -2){
        list->size--;
    }
    list->data[index] = -2;
}

void updateElement(struct LinkedList *list, int index, int value) {
    list->data[index] = value;
}

int getNextIndex(struct LinkedList *list, int index){
    int curr_index = index+1;
    while (list->data[curr_index] == -2 && curr_index < list->capacity){
        curr_index++;
    }
    if (curr_index >= list->capacity){
        return -1;
    }
    return curr_index;
}

int getPrevIndex(struct LinkedList *list, int index){
    int curr_index = index-1;
    while (list->data[curr_index] == -2 && curr_index >= 0){
        curr_index--;
    }
    if (curr_index < 0){
        return -1;
    }
    return curr_index;
}

int getNextElement(struct LinkedList *list, int index){
    int next_index = getNextIndex(list, index);
    if (next_index != -1){
        return list->data[next_index];
    }
    return -1;
}

int getPrevElement(struct LinkedList *list, int index){
    int prev_index = getPrevIndex(list, index);
    if (prev_index != -1){
        return list->data[prev_index];
    }
    return -1;
}

// Function to display the doubly linked list
void displayList(struct LinkedList &list, int raw) {
    int curr = getNextIndex(&list, -1);
    while (curr != -1){
        printf("%d, ", list.data[curr]);
        if (raw){
            curr++;
            if (curr >= list.capacity){
                curr = -1;
            }
        }
        else{
            curr = getNextIndex(&list, curr);
        }
    }
    printf("\n");
}

void shuffleArray(int arr[], int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1); // Generate a random index from 0 to i
        // Swap arr[i] and arr[j]
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

struct HeapNode {
    int key; // size of the set stored at this pair
    uint64_t pair_index; // index of the pair
};

typedef struct HeapNode heapNode;

struct Heap {
    heapNode* arr; // pointer to the data
    std::unordered_map<uint64_t, int> pairPositions;
    std::unordered_map<uint64_t, std::unique_ptr<std::unordered_set<int>>> pairSets;
    int size;
    int capacity;
    int vocabSize;
};

typedef struct Heap heap;
void heapify(heap* h, int index);
void increaseKey(heap* h, int position, int newKey);
void insert(heap* h, int key, uint64_t pair_index);

void swap(heap *h, int pos1, int pos2){
    // swap two elements in the heap
    heapNode temp = h->arr[pos1];
    h->pairPositions[temp.pair_index] = pos2;
    h->pairPositions[h->arr[pos2].pair_index] = pos1;
    h->arr[pos1] = h->arr[pos2];
    h->arr[pos2] = temp;
}

void bubbleUp(heap* h, int index)
{
    int parent = (index - 1) / 2;
    if (h->arr[parent].key < h->arr[index].key) {
        swap(h, parent, index);
        bubbleUp(h, parent);
    }
}
 
void heapify(heap* h, int index)
{
    int left = index * 2 + 1;
    int right = index * 2 + 2;
    int max = index;
    if (left >= h->size || left < 0)
        left = -1;
    if (right >= h->size || right < 0)
        right = -1;
    if (left != -1 && h->arr[left].key > h->arr[index].key)
        max = left;
    if (right != -1 && h->arr[right].key > h->arr[max].key)
        max = right;
    if (max != index) {
        swap(h, max, index);
        heapify(h, max);
    }
}

void deleteEl(heap *h, int position, uint64_t pair_index){
    swap(h, position, h->size-1);
    h->size--;
    if (h->arr[h->size].key > h->arr[position].key){
        heapify(h, position); // key decreased
    }
    else{
        bubbleUp(h, position); // key increased
    }
    h->pairPositions.erase(pair_index);
    h->pairSets.erase(pair_index);
}

// returns the maximum pair as integer:
uint64_t extractMaxPair(heap* h)
{
    if (h->size == 0) {
        return -1; // Return NULL to indicate an empty heap
    }
    uint64_t result = h->arr[0].pair_index;
    // printf("max value: %f := ", h->arr[0].key);
    swap(h, 0, h->size-1);
    h->size--;
    heapify(h, 0);
    // deleteEl(h, 0, result);
    return result;
}
 
// add new pair to the heap
void insert(heap* h, int key, uint64_t pair_index)
{
    // key := position inside the array, pair_index := index of the pair
    if (h->size < h->capacity) {
        h->arr[h->size].key = key;
        h->arr[h->size].pair_index = pair_index;
        h->pairSets[pair_index] = std::make_unique<std::unordered_set<int>>();
        h->pairPositions[pair_index] = h->size;
        bubbleUp(h, h->size);
        h->size++;
    }
    else{
        printf("ERROR on insertion:(\n");
    }
}

void decreaseKey(heap* h, int position, int newKey, int pair_index){
    if (newKey == 0){
        deleteEl(h, position, pair_index);
    }
    else{
        h->arr[position].key = newKey;
        heapify(h, position);
    }
}

void increaseKey(heap* h, int position, int newKey){
    h->arr[position].key = newKey;
    bubbleUp(h, position);
}

void printHeap(heap* h)
{
    printf("Heap:\n");
    for (int i = 0; i < h->size; i++) {
        int m_len = h->vocabSize;
        int pair_1 = (int)(h->arr[i].pair_index / m_len);
        int pair_2 = (h->arr[i].pair_index) % m_len;
        printf("Key: %d, Pair: (%d, %d) ", h->arr[i].key, pair_1, pair_2);
        printf("Positions: ");
        uint64_t pair_index = h->arr[i].pair_index;
        for (auto& pos : *(h->pairSets[pair_index])) {
            printf("%d ", pos);
        }
        printf("\n");
    }
    // printf("\nPair indices:\n");
    // for (int i = 0; i < h->size; i++) {
    //     printf("Position in heap: %d\n", h->pairPositions[h->arr[i].pair_index]);
    // }
}

void freeHeap(heap *hp){
    free(hp->arr);
    hp->pairPositions.clear();
    hp->pairSets.clear();
}

// **********************************
// Main functions:
// **********************************

/**
 * @brief Creates a heap object and returns it by value
 *
 * Creates a heap object that is used inside the train function.
 *
 * @param vocab_size The size of the vocabulary.
 * @param ids the list containing the text data.
 * @return A heap object representing the created heap.
 */
heap createHeap(int vocab_size, struct LinkedList &ids, int init_tokens)
{
    heap h; // create a heap object on the stack
    int capacity = std::min((uint64_t)vocab_size*vocab_size, (uint64_t)ids.size);
    // maybe replace with array, but then a lot of memory
    h.pairSets = std::unordered_map<uint64_t, std::unique_ptr<std::unordered_set<int>>>();
    
    {
        // performance improvement: store positions in vector first and then add to sets
        std::vector<std::vector<int>> pair_positions_fast(init_tokens * init_tokens);
        // count all pairs initially:
        for (int i = 0; i < ids.size-1; i++){
            int first_el = ids.data[i];
            int second_el = ids.data[i + 1];
            int index = first_el*init_tokens + second_el;
            if (first_el != 0 && second_el != 0){
                pair_positions_fast[index].push_back(i);
            }
        }
        for (size_t i = 0; i < init_tokens * init_tokens; i++)
        {
            int tok_1 = (int)i/init_tokens;
            int tok_2 = i%init_tokens;
            uint64_t real_index = (uint64_t)tok_1*vocab_size + tok_2;
            if (pair_positions_fast[i].size() > 0){
                h.pairSets[real_index] = std::make_unique<std::unordered_set<int>>(pair_positions_fast[i].size());
                // insertion is much faster here because the array of the hashmap remains loaded into cache:
                for (size_t j = 0; j < pair_positions_fast[i].size(); j++)
                {
                    h.pairSets[real_index]->insert(pair_positions_fast[i][j]);
                }
            }
        }
    }

    h.pairPositions = std::unordered_map<uint64_t, int>(); // initialize the map with the constructor
    h.size = 0;
    h.capacity = capacity;
    h.vocabSize = vocab_size;
    h.arr = (heapNode*)malloc(capacity * sizeof(heapNode));
 
    if (h.arr == NULL) {
        printf("Memory error");
        return h;
    }
    // build inital heap:
    int count = 0;
    for (const auto& pair : h.pairSets) {
        uint64_t key = pair.first;
        int size = pair.second->size();
        h.arr[count].key = size;
        h.arr[count].pair_index = key;
        h.pairPositions[key] = count;
        count++;
    }

    // bottom up heap construction (takes O(n) time):
    h.size = count;
    count = (h.size - 2) / 2;
    while (count >= 0) {
        heapify(&h, count);
        count--;
    }
    return h; // return the heap object by value
}

void RemovePosition(heap *h, int index, int tok_1, int tok_2, uint64_t disallowed){
    // index denotes the start position of the pair inside the linked list
    // the disallowed token only ensures that the pair is only removed when
    // it is unequal to the disallowed argument. This is needed for the repeating
    // tokens problem.
    if (tok_1 <= 0 || tok_2 <= 0){ // ignore if separated by 0 or out of bound
        return;
    }
    // removes stored pair position and updates the heap
    uint64_t pair_index = (uint64_t)tok_1 * h->vocabSize + tok_2;
    if (pair_index == disallowed){
        return;
    }
    auto& pairSet = h->pairSets[pair_index];
    pairSet->erase(index);
    int new_len = pairSet->size();
    if (new_len == 0){
        int c = 0;
    }
    decreaseKey(h, h->pairPositions[pair_index], new_len, pair_index);
}

void AddPosition(heap *h, int index, int tok_1, int tok_2){
    // index denotes the start position of the pair inside the linked list
    if (tok_1 <= 0 || tok_2 <= 0){ // ignore if separated by 0 or out of bound
        return;
    }
    // adds a new pair position and updates the heap
    uint64_t pair_index = (uint64_t) tok_1 * h->vocabSize + tok_2;
    if (h->pairSets.find(pair_index) == h->pairSets.end()){
        // if not added yet (unseen pair), add it to the heap
        insert(h, 0, pair_index);
    }
    auto& pairSet = h->pairSets[pair_index];
    pairSet->insert(index);
    int new_len = pairSet->size();
    increaseKey(h, h->pairPositions[pair_index], new_len);
}

// used as return values. Stores every nessesary information:
struct Token {
    int token_id; // the id of the token
    int first_id; // the first element of the pair
    int second_id; // the second element of the pair
    int token_list_len; // the length of the corresponding tokens
    int* token_list; // the corresponding tokens
};

extern "C"{
/**
 * @brief Main algorithm for training tokenization model.
 *
 * This function implements the main algorithm for training a tokenization model
 * based on the provided integer IDs. https://aclanthology.org/2023.findings-acl.38.pdf
 *
 * @param ids An array containing the input IDs.
 * @param num_ids The number of IDs in the 'ids' array.
 * @param num_tokens the number of tokens we want to train in total.
 * @param init_tokens The number of initial tokens to start with.
 * @return A pointer to a struct Token representing the trained model.
 *         Note: Memory for the struct Token and its contents should be managed appropriately.
 */
struct Token* train(int* ids, int num_ids, int num_tokens, int init_tokens) {
    // build vocab
    if (init_tokens <= 0){
        init_tokens = 1;
    }
    struct Token* vocab = (struct Token*)malloc(num_tokens * sizeof(struct Token));
    // build initial vocab:
    for (int i = 0; i < num_tokens; i++) {
        int a_len = 1;
        vocab[i].token_id = i;
        vocab[i].first_id = i;
        vocab[i].second_id = i;
        vocab[i].token_list_len = a_len;
        vocab[i].token_list = (int*)malloc(a_len * sizeof(int));
        for (int j = 0; j < a_len; j++) {
            vocab[i].token_list[j] = i;
        }
    }

    // build inital heap:
    struct LinkedList list = createLinkedList(ids, num_ids);
    heap h = createHeap(num_tokens, list, init_tokens);

    // number of merges we still need:
    int total_merges = num_tokens-init_tokens;

    // main algorithm:
    for (size_t i = 0; i < total_merges; i++)
    {
        // displayList(list, 0);
        // printHeap(&h);
        // get max pair:

        uint64_t best_pair = extractMaxPair(&h);
        if (best_pair == -1){
            vocab[0].token_id = -1; // mark to catch error in python
            break;
        }
        int max_pair_1 = (int)(best_pair / h.vocabSize);
        int max_pair_2 = best_pair % h.vocabSize;
        // store new token in the final output:
        int new_token_id = init_tokens + i;
        vocab[new_token_id].token_id = new_token_id;
        vocab[new_token_id].first_id = max_pair_1;
        vocab[new_token_id].second_id = max_pair_2;
        int pair_1_len = vocab[max_pair_1].token_list_len;
        int pair_2_len = vocab[max_pair_2].token_list_len;
        int new_tokens_len = pair_1_len + pair_2_len;
        vocab[new_token_id].token_list_len = new_tokens_len;
        vocab[new_token_id].token_list = (int*)malloc(new_tokens_len * sizeof(int));
        // create new list of tokens by merging the old ones:
        for (size_t i = 0; i < pair_1_len; i++)
        {
            vocab[new_token_id].token_list[i] = vocab[max_pair_1].token_list[i];
        }
        for (size_t i = 0; i < pair_2_len; i++)
        {
            vocab[new_token_id].token_list[i+pair_1_len] = vocab[max_pair_2].token_list[i];
        }
        auto& pairSet = h.pairSets[best_pair];
        std::vector<int> sortedPairSet;
        int c = pairSet->size();
        sortedPairSet.reserve(pairSet->size());
        sortedPairSet.insert(sortedPairSet.end(), pairSet->begin(), pairSet->end());
        std::sort(sortedPairSet.begin(), sortedPairSet.end());
        h.pairPositions.erase(best_pair);
        h.pairSets.erase(best_pair);

        int last_added_index = -1;
        for (auto& pos: sortedPairSet) {
            if (pos == last_added_index){ // needed for repeating tokens
                continue;
            }
            // pos = position inside the linked list
            int w1 = list.data[pos];
            int w1_prev_ind = getPrevIndex(&list, pos);
            int w1_prev = -1;
            if (w1_prev_ind != -1){
                w1_prev = list.data[w1_prev_ind];
            }
            int w2_ind = getNextIndex(&list, pos);
            int w2 = list.data[w2_ind]; // has to exist, otherwise we made a mistake
            int w2_next = getNextElement(&list, w2_ind);
            RemovePosition(&h, w1_prev_ind, w1_prev, w1, best_pair);
            RemovePosition(&h, w2_ind, w2, w2_next, best_pair);
            int next_id = getNextIndex(&list, pos);
            // merge tokens to create new token:
            deleteElement(&list, next_id);
            updateElement(&list, pos, new_token_id);
            // add new pairs:
            AddPosition(&h, pos, new_token_id, w2_next);
            AddPosition(&h, w1_prev_ind, w1_prev, new_token_id);
            last_added_index = next_id;
        }
    }
    freeHeap(&h);
    return vocab;
}
}

void printVector(const std::vector<uint16_t>& vec) {
    for (int value : vec) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

struct tokenStat{
    int64_t pair_id;
    uint32_t tok_id;
};

void printVector(const std::vector<struct tokenStat>& vec) {
    for (auto &value : vec) {
        std::cout << value.tok_id << " ";
    }
    std::cout << std::endl;
}

void _tokenizeChunk(std::vector<uint32_t> &ids, std::unordered_map<int64_t, uint32_t> &pair_to_tok, int vocab_size){
    // tokenizes a chunk in-place
    std::vector<struct tokenStat> stats; // Vector of struct stat
    stats.reserve(ids.size());
    for (size_t i = 0; i < ids.size()-1; i++)
    {
        uint32_t t_1 = ids[i];
        uint32_t t_2 = ids[i+1];
        int64_t key = (int64_t)t_1 * vocab_size + t_2;
        if (pair_to_tok.find(key) != pair_to_tok.end()){
            // Construct a struct stat and push it into the vector
            stats.push_back({key, pair_to_tok[key]});
        }
    }

    // merge and recalculate stats while merging
    while (stats.size() > 0){
        // find the first pair that has to be swapped:
        uint32_t min_tok_id = -1;
        int64_t min_pair_id = -1;
        for (size_t i = 0; i < stats.size(); i++)
        {
            if (stats[i].tok_id < min_tok_id || min_tok_id == -1){
                min_tok_id = stats[i].tok_id;
                min_pair_id = stats[i].pair_id;
            }
        }
        // delete every occurrence of the min pair:
        for (size_t i = 0; i < stats.size(); i++)
        {
            if (stats[i].tok_id == min_tok_id){
                // delete element:
                stats[i].tok_id = stats[stats.size()-1].tok_id;
                stats[i].pair_id = stats[stats.size()-1].pair_id;
                stats.pop_back(); // resize stats
                i--; // last element might also be min_tok_id
            }
        }
        uint32_t tok_1 = min_pair_id / vocab_size;
        uint32_t tok_2 = min_pair_id % vocab_size;
        // merge every occurrence:
        int i = 0;
        int curr_append = 0;
        uint32_t prev = -1;
        while (i < ids.size()){
            if (ids[i] == tok_1 && i < ids.size() - 1 && ids[i+1] == tok_2){
                ids[curr_append] = min_tok_id;
                i += 2;
                if (prev != -1){
                    int64_t pair_id =(int64_t)prev*vocab_size + ids[curr_append];
                    if (pair_to_tok.find(pair_id) != pair_to_tok.end()){
                        stats.push_back({pair_id, pair_to_tok[pair_id]});
                    }
                }
            }
            else{
                ids[curr_append] = ids[i];
                i += 1;
            }
            if (prev == min_tok_id && ids[curr_append] != min_tok_id){
                int64_t pair_id = (int64_t)prev*vocab_size + ids[curr_append];
                if (pair_to_tok.find(pair_id) != pair_to_tok.end()){
                    stats.push_back({pair_id, pair_to_tok[pair_id]});
                }
            }
            prev = ids[curr_append];
            curr_append++;
        }
        int to_pop = ids.size()-curr_append;
        for (size_t i = 0; i < to_pop; i++)
        {
            ids.pop_back();
        }
    }
}

struct tokenizeResult
{
    uint32_t *ids;
    int ids_size;
};


extern "C"{
/**
 * @brief Tokenizes text given by an array of IDs and uses token pairs to decode the text.
 *
 * This function tokenizes text represented by an array of integer IDs using the provided token pairs
 * to decode the text.
 *
 * @param ids An array containing the text to be tokenized.
 * @param num_ids The length of the 'ids' array.
 * @param token_pairs An array containing token pairs obtained from the train function.
 * @param token_pairs_count The number of token pairs in the 'token_pairs' array.
 * @param vocab_size The number of tokens used in the tokenizer.
 * @return A pointer to an array of integers representing the tokenized text.
 *         This array needs to be freed after use.
 */
    struct tokenizeResult tokenize(uint8_t *ids, int num_ids, int *splits, int len_splits, int64_t *token_pairs, int token_pairs_count, int init_tokens, int num_threads){
        int vocab_size = token_pairs_count+init_tokens;
        auto start = std::chrono::steady_clock::now(); // Record start time
        // splits denote the places where the string got splitted by regex
        std::vector<std::vector<uint32_t>> splitted;
        splitted.reserve(len_splits);
        for (size_t i = 0; i < len_splits-1; i++)
        {
            int curr = splits[i];
            int next = splits[i+1];
            std::vector<uint32_t> chunk;
            chunk.reserve(next-curr);
            for (size_t j = curr; j < next; j++)
            {
                chunk.emplace_back((uint32_t)ids[j]);
            }
            splitted.emplace_back(chunk);
        }
        
        std::unordered_map<int64_t, uint32_t> pair_to_token(token_pairs_count);
        for (size_t i = 0; i < token_pairs_count; i++)
        {
            pair_to_token[token_pairs[i]] = (uint32_t)(init_tokens + i);
        }
        
        // Vector to store threads
        std::vector<std::thread> threads;

        // Function to be executed by each thread
        auto tokenizeChunksThread = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                _tokenizeChunk(splitted[i], pair_to_token, vocab_size);
            }
        };

        // Calculate chunk size based on the number of threads
        size_t chunkSize = splitted.size() / num_threads;

        // Create threads
        for (size_t i = 0; i < num_threads - 1; i++) {
            threads.emplace_back(tokenizeChunksThread, i * chunkSize, (i + 1) * chunkSize);
        }
        // Main thread handles remaining chunks
        for (size_t i = (num_threads - 1) * chunkSize; i < splitted.size(); i++) {
            _tokenizeChunk(splitted[i], pair_to_token, vocab_size);
        }
        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }
        
        int total_size = 0;
        for (size_t i = 0; i < splitted.size(); i++)
        {
            total_size += splitted[i].size();
        }
        
        uint32_t *result = (uint32_t*)malloc(sizeof(uint32_t)*total_size);
        int c = 0;
        for (size_t i = 0; i < splitted.size(); i++)
        {
            for (size_t j = 0; j < splitted[i].size(); j++)
            {
                result[c] = splitted[i][j];
                c++;
            }
        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "Elapsed time inside tokenize function: " << duration.count() << " milliseconds" << std::endl;
        return tokenizeResult{result, total_size};
    }
}

// int main() {
//     srand(time(NULL)); // Seed for random number generation
//     auto start = std::chrono::steady_clock::now(); // Record start time

//     // Array with specified numbers
//     int ids[] = {45, 115, 101, 101, 107, 101, 114, 0, 44, 0, 32, 116, 104, 101, 0, 32, 116, 101, 97, 109, 0, 32, 117, 115, 101, 100, 0, 32, 97, 0, 32, 115, 116, 105, 99, 107, 0, 32, 111, 110, 0, 32, 115, 101, 116, 0, 32, 97, 115, 0, 32, 114, 101, 102, 101, 114, 101, 110, 99, 101, 0};
//     int num_ids = sizeof(ids) / sizeof(ids[0]); // Calculate number of elements in the array

//     int num_tokens = 260; // Vocab size
//     int init_tokens = 256; // Init size
//     struct Token* vocab = train(ids, num_ids, num_tokens, init_tokens);

//     // for (int i = 0; i < num_tokens; i++) {
//     //     printf("(%d, %d) => %d, Token List Length: %d, Token List: ",
//     //         vocab[i].first_id, vocab[i].second_id, vocab[i].token_id, vocab[i].token_list_len);
//     //     for (int j = 0; j < vocab[i].token_list_len; j++) {
//     //         printf("%d ", vocab[i].token_list[j]);
//     //     }
//     //     printf("\n");
//     //     free(vocab[i].token_list);
//     // }
//     free(vocab);
//     auto end = std::chrono::steady_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//     std::cout << "Elapsed time inside tokenize function: " << duration.count() << " milliseconds" << std::endl;
//     return 0;
// }

int main() {
    srand(time(NULL)); // Seed for random number generation
    auto start = std::chrono::steady_clock::now(); // Record start time

    // Generate an array with random numbers of size 10000
    int num_elements = 100000000;
    int init_tokens = 256;
    int* ids = new int[num_elements];
    for (int i = 0; i < num_elements; ++i) {
        ids[i] = rand() % init_tokens; // Random numbers between 0 and 99
    }

    int num_tokens = 10000; // Vocab size
    struct Token* vocab = train(ids, num_elements, num_tokens, init_tokens);

    for (int i = 0; i < num_tokens; i++) {
        //std::cout << "(" << vocab[i].first_id << ", " << vocab[i].second_id << ") => " << vocab[i].token_id << ", Token List Length: " << vocab[i].token_list_len << ", Token List: ";
        for (int j = 0; j < vocab[i].token_list_len; j++) {
           // std::cout << vocab[i].token_list[j] << " ";
        }
        //std::cout << std::endl;
        delete[] vocab[i].token_list;
    }

    delete[] ids;
    delete[] vocab;

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time inside tokenize function: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}

// int main() {
//     // little test:
//     int vocab_size = 13;
//     int init_tokens = 6;
//     std::vector<uint8_t> test = {0, 1, 2, 3, 3, 4, 5, 5};
//     std::vector<int> splits = {0, 8};
//     std::vector<int64_t> token_pairs = {
//         2*vocab_size + 3, // 6
//         4*vocab_size + 5, // 7
//         0*vocab_size + 1, // 8
//         7*vocab_size + 5, // 9
//         6*vocab_size + 3, // 10
//         8*vocab_size + 10, // 11
//         11*vocab_size + 9, // 12
//     };
//     struct tokenizeResult res = tokenize(&test[0], test.size(), &splits[0], splits.size(), &token_pairs[0], vocab_size-init_tokens, init_tokens, 1);
//     printf("Result:"); // should print 12
//     for (size_t i = 0; i < res.ids_size; i++)
//     {
//         printf("%d ", res.ids[i]);
//     }
//     return 0;
// }

// int main() {
//     srand(time(NULL)); // Seed for random number generation
//     int num_ids = 10000; // chunk size
//     int *ids = (int *)malloc(num_ids * sizeof(int));

//     // Fill array with random numbers from 1 to 255
//     for (int i = 0; i < num_ids; i++) {
//         ids[i] = rand() % 255 + 1;
//     }

//     int num_tokens = 10000;
//     int init_tokens = 256;
//     struct Token* vocab = train(ids, num_ids, num_tokens, init_tokens);

//     // for (int i = 0; i < num_tokens; i++) {
//     //     printf("Token ID: %d, First ID: %d, Second ID: %d, Token List Length: %d, Token List: ",
//     //            vocab[i].token_id, vocab[i].first_id, vocab[i].second_id, vocab[i].token_list_len);
//     //     for (int j = 0; j < vocab[i].token_list_len; j++) {
//     //         printf("%d ", vocab[i].token_list[j]);
//     //     }
//     //     printf("\n");
//     //     free(vocab[i].token_list);
//     // }
//     free(vocab);
//     free(ids); // Free dynamically allocated memory for ids array
//     return 0;
// }