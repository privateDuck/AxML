#pragma once

#ifndef AXML_MEM_MGR_HPP
#define AXML_MEM_MGR_HPP

#include <memory_resource>

namespace AxML {

    namespace ll {
        inline std::pmr::memory_resource* get_tls_pool() {
            thread_local std::pmr::unsynchronized_pool_resource tls_pool;
            return &tls_pool;
        }

        template<typename T>
        struct TlAllocator {
            using value_type = T;
            TlAllocator() noexcept = default;

            template<class U>
            explicit constexpr TlAllocator(const TlAllocator<U>&) noexcept {}

            T* allocate(const std::size_t n) {
                return static_cast<T*>(get_tls_pool()->allocate(n * sizeof(T), alignof(T)));
            }

            void deallocate(T* p, const std::size_t n) {
                get_tls_pool()->deallocate(p, n * sizeof(T), alignof(T));
            }
        };

        template <class T, class U>
        bool operator ==(const TlAllocator<T>&, const TlAllocator<U>&) { return true; }
        template <class T, class U>
        bool operator !=(const TlAllocator<T>&, const TlAllocator<U>&) { return false; }

        template <typename T>
        using tl_vec = std::vector<T, TlAllocator<T>>;
    }


    using SyncResource = std::pmr::synchronized_pool_resource;
    using ASyncResource = std::pmr::unsynchronized_pool_resource;
}

#endif //AXML_MEM_MGR_HPP