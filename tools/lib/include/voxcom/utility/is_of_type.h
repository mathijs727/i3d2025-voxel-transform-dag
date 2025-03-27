#pragma once
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace voxcom {

template <typename>
struct is_std_vector : std::false_type {
};
template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {
};

template <typename>
struct is_std_variant : std::false_type {
};
template <typename... Ts>
struct is_std_variant<std::variant<Ts...>> : std::true_type {
};

template <typename>
struct is_std_optional : std::false_type {
};
template <typename T>
struct is_std_optional<std::optional<T>> : std::true_type {
};
template <typename T>
constexpr bool is_std_optional_v = is_std_optional<T>::value;

template <typename>
struct std_optional_type {
};
template <typename T>
struct std_optional_type<std::optional<T>> {
    using type = T;
};
template <typename T>
using std_optional_type_t = typename std_optional_type<T>::type;

template <typename>
struct is_std_unique_ptr : std::false_type {
};
template <typename T>
struct is_std_unique_ptr<std::unique_ptr<T>> : std::true_type {
};
template <typename T>
constexpr bool is_std_unique_ptr_v = is_std_unique_ptr<T>::value;

}
