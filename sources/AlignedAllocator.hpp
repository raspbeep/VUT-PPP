/**
 * @file    AlignedAllocator.hpp
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *          David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *
 * @date    2024-02-22
 */

#ifndef ALIGNED_ALLOCATOR_HPP
#define ALIGNED_ALLOCATOR_HPP

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

/**
 * @brief An allocator that allocates memory with a predefined alignment.
 * @tparam T The type of the elements to allocate.
 */
template<typename T = void>
struct AlignedAllocator
{
  /// @brief The type of the deleter used to deallocate memory. Useful for smart pointers.
  struct Deleter;

  /// @brief Required type for the allocator concept.
  using value_type = T;

  /// @brief Default constructor.
  AlignedAllocator() = default;

  /**
   * @brief Copy constructor.
   * @tparam U Other's value type.
   * @param other The other allocator to copy.
   */
  template<typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U>& other) noexcept;

  /**
   * @brief Move constructor.
   * @tparam U Other's value type.
   * @param other The other allocator to move.
   */
  template<typename U>
  constexpr AlignedAllocator(AlignedAllocator<U>&& other) noexcept;

  /// @brief Destructor.
  ~AlignedAllocator() = default;

  /**
   * @brief Copy assignment operator.
   * @tparam U Other's value type.
   * @param other The other allocator to copy.
   * @return A reference to this.
   */
  template<typename U>
  AlignedAllocator& operator=(const AlignedAllocator<U>& other) noexcept;

  /**
   * @brief Move assignment operator.
   * @tparam U Other's value type.
   * @param other The other allocator to move.
   * @return A reference to this.
   */
  template<typename U>
  AlignedAllocator& operator=(AlignedAllocator<U>&& other) noexcept;

  /**
   * @brief Allocate memory for n elements of type T.
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  [[nodiscard]] T* allocate(std::size_t n);

  /**
   * @brief Deallocate memory previously allocated with allocate.
   * @param p A pointer to the memory to deallocate.
   * @param n The number of elements previously allocated (ignored, but required by the allocator concept).
   */
  void deallocate(T* p, [[maybe_unused]] std::size_t n = {}) noexcept;

  /// @brief The alignment of the allocated memory.
  static constexpr std::size_t alignment{64};
};

/**
 * @brief The type of the deleter used to deallocate memory. Useful for smart pointers.
 * @tparam T The type of the elements to deallocate.
 */
template<typename T>
struct AlignedAllocator<T>::Deleter
{
  /**
   * @brief Deallocate memory previously allocated with allocate.
   * @param p A pointer to the memory to deallocate.
   */
  void operator()(T* p) const noexcept;
};

/**
 * @brief Compare two allocators for equality.
 * @tparam T The type of lhs allocator.
 * @tparam U The type of rhs allocator.
 * @param lhs The first allocator to compare.
 * @param rhs The second allocator to compare.
 * @return True if the allocators are equal, false otherwise.
 */
template<typename T, typename U>
constexpr bool operator==(const AlignedAllocator<T>& lhs, const AlignedAllocator<U>& rhs) noexcept;

/**
 * @brief Compare two allocators for inequality.
 * @tparam T The type of lhs allocator.
 * @tparam U The type of rhs allocator.
 * @param lhs The first allocator to compare.
 * @param rhs The second allocator to compare.
 * @return True if the allocators are not equal, false otherwise.
 */
template<typename T, typename U>
constexpr bool operator!=(const AlignedAllocator<T>& lhs, const AlignedAllocator<U>& rhs) noexcept;

template<typename T>
template<typename U>
constexpr AlignedAllocator<T>::AlignedAllocator(const AlignedAllocator<U>&) noexcept
{}

template<typename T>
template<typename U>
constexpr AlignedAllocator<T>::AlignedAllocator(AlignedAllocator<U>&&) noexcept
{}

template<typename T>
template<typename U>
AlignedAllocator<T>& AlignedAllocator<T>::operator=(const AlignedAllocator<U>&) noexcept
{
  return *this;
}

template<typename T>
template<typename U>
AlignedAllocator<T>& AlignedAllocator<T>::operator=(AlignedAllocator<U>&&) noexcept
{
  return *this;
}

template<typename T>
auto AlignedAllocator<T>::allocate(std::size_t n) -> T*
{
  T* ptr{};
#ifdef _WIN32
  ptr = static_cast<T*>(_aligned_malloc(n * sizeof(T), alignment));
#else
  ptr = static_cast<T*>(aligned_alloc(alignment, n * sizeof(T)));
#endif /* _WIN32 */

  if (ptr == nullptr)
  {
    throw std::bad_alloc{};
  }

  return ptr;
}

template<typename T>
void AlignedAllocator<T>::deallocate(T* p, std::size_t) noexcept
{
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif /* _WIN32 */
}

template<typename T>
void AlignedAllocator<T>::Deleter::operator()(T* p) const noexcept
{
  AlignedAllocator<T>{}.deallocate(p);
}

template<typename T, typename U>
constexpr bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept
{
  return true;
}

template<typename T, typename U>
constexpr bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept
{
  return false;
}

#endif /* ALIGNED_ALLOCATOR_HPP */
