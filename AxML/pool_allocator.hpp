#pragma once
#ifndef AXML_POOL_ALLOCATOR_HPP
#define AXML_POOL_ALLOCATOR_HPP

#include <memory_resource>

struct PooledResource {
    std::vector<std::byte> memory;
    std::pmr::monotonic_buffer_resource mbr;
    std::pmr::unsynchronized_pool_resource async_pool;

    explicit PooledResource(const size_t backing_pool_size)
        : memory(backing_pool_size),
          mbr(memory.data(), memory.size(), std::pmr::new_delete_resource()),
          async_pool(&mbr) {}

    std::pmr::unsynchronized_pool_resource* get_pool_resource() { return &async_pool; }
};

#endif //AXML_POOL_ALLOCATOR_HPP