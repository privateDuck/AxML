#pragma once

#ifndef AXML_MEM_MGR_HPP
#define AXML_MEM_MGR_HPP

#include <memory_resource>
#include <unordered_map>

namespace AxML::ll {

    inline std::pmr::memory_resource* get_tls_pool() {
        thread_local std::pmr::unsynchronized_pool_resource tls_pool;
        return &tls_pool;
    }

    template<typename T>
    struct TlsAllocator {
        using value_type = T;
        TlsAllocator() noexcept = default;

        template<class U>
        explicit constexpr TlsAllocator(const TlsAllocator<U>&) noexcept {}

        T* allocate(const std::size_t n) {
            return static_cast<T*>(get_tls_pool()->allocate(n * sizeof(T), alignof(T)));
        }

        void deallocate(T* p, const std::size_t n) {
            get_tls_pool()->deallocate(p, n * sizeof(T), alignof(T));
        }
    };

    template <class T, class U>
    bool operator ==(const TlsAllocator<T>&, const TlsAllocator<U>&) { return true; }
    template <class T, class U>
    bool operator !=(const TlsAllocator<T>&, const TlsAllocator<U>&) { return false; }

    template <typename T>
    using tl_vec = std::vector<T, TlsAllocator<T>>;
}

#endif //AXML_MEM_MGR_HPP