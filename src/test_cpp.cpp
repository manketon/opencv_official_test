#include "test_cpp.h"
#include <iostream>
#include <vector>
#define USING_CUSTOM_ALLOCATOR
// 自定义分配器（在类型std::allocator基础上修改所得）
template <class _Ty>
class CustomAllocator {
public:
	static_assert(!std::is_const_v<_Ty>, "The C++ Standard forbids containers of const elements "
		"because allocator<const T> is ill-formed.");

	using _From_primary = CustomAllocator;

	using value_type = _Ty;

#if _HAS_DEPRECATED_ALLOCATOR_MEMBERS
	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS typedef _Ty* pointer;
	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS typedef const _Ty* const_pointer;

	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS typedef _Ty& reference;
	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS typedef const _Ty& const_reference;
#endif // _HAS_DEPRECATED_ALLOCATOR_MEMBERS

	using size_type = size_t;
	using difference_type = ptrdiff_t;

	using propagate_on_container_move_assignment = std::true_type;

#if _HAS_DEPRECATED_ALLOCATOR_MEMBERS
	using is_always_equal _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS = std::true_type;

	template <class _Other>
	struct _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS rebind {
		using other = CustomAllocator<_Other>;
	};

	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD _Ty* address(_Ty& _Val) const noexcept {
		return _STD addressof(_Val);
	}

	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD const _Ty* address(const _Ty& _Val) const noexcept {
		return _STD addressof(_Val);
	}
#endif // _HAS_DEPRECATED_ALLOCATOR_MEMBERS

	constexpr CustomAllocator() noexcept {}

	constexpr CustomAllocator(const CustomAllocator&) noexcept = default;
	template <class _Other>
	constexpr CustomAllocator(const CustomAllocator<_Other>&) noexcept {}
	_CONSTEXPR20_DYNALLOC ~CustomAllocator() = default;
	_CONSTEXPR20_DYNALLOC CustomAllocator& operator=(const CustomAllocator&) = default;
	//__declspec(allocator)为Microsoft 专用，声明说明符可应用于自定义内存分配函数，以通过 Windows 事件跟踪 (ETW) 使分配可见
#ifdef USING_CUSTOM_ALLOCATOR
	_NODISCARD _CONSTEXPR20_DYNALLOC __declspec(allocator) _Ty* allocate(_CRT_GUARDOVERFLOW const size_t _Count)
	{
		std::cout << __FUNCTION__ << " | line:" << __LINE__ << ", Count:" << _Count << std::endl;
		void* p = ::malloc(std::_Get_size_of_n<sizeof(_Ty)>(_Count));
		return static_cast<_Ty*>(p);
	}

	_CONSTEXPR20_DYNALLOC void deallocate(_Ty* const _Ptr, const size_t _Count)
	{
		std::cout << __FUNCTION__ << " | line:" << __LINE__ << std::endl;
		if (_Ptr)
		{
			::free(_Ptr);
		}
	}
#else //与std::allocator中逻辑等价
	_CONSTEXPR20_DYNALLOC void deallocate(_Ty* const _Ptr, const size_t _Count) {
		// no overflow check on the following multiply; we assume _Allocate did that check
		std::_Deallocate<std::_New_alignof<_Ty>>(_Ptr, sizeof(_Ty) * _Count);
	}

	_NODISCARD _CONSTEXPR20_DYNALLOC __declspec(allocator)_Ty* allocate(_CRT_GUARDOVERFLOW const size_t _Count) {
		return static_cast<_Ty*>(std::_Allocate<std::_New_alignof<_Ty>>(std::_Get_size_of_n<sizeof(_Ty)>(_Count)));
	}
#endif
#if _HAS_DEPRECATED_ALLOCATOR_MEMBERS
	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD __declspec(allocator) _Ty* allocate(
		_CRT_GUARDOVERFLOW const size_t _Count, const void*) {
		std::cout << __FUNCTION__ << " | line:" << __LINE__ << std::endl;
		return allocate(_Count);
	}

	template <class _Objty, class... _Types>
	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS void construct(_Objty* const _Ptr, _Types&&... _Args) {
		std::cout << __FUNCTION__ << " | line:" << __LINE__ << std::endl;
		::new (std::_Voidify_iter(_Ptr)) _Objty(_STD forward<_Types>(_Args)...); //在已经分配好的内存上构造对象（不新分配内存）
	}

	template <class _Uty>
	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS void destroy(_Uty* const _Ptr) {
		std::cout << __FUNCTION__ << " | line:" << __LINE__ << std::endl;
		_Ptr->~_Uty();
	}

	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD size_t max_size() const noexcept {
		std::cout << __FUNCTION__ << " | line:" << __LINE__ << std::endl;
		return static_cast<size_t>(-1) / sizeof(_Ty);
	}
#endif // _HAS_DEPRECATED_ALLOCATOR_MEMBERS
};

int test_CustomAllocator(std::string& str_err_reason)
{
	// 使用自定义分配器初始化vector
	std::vector<int, CustomAllocator<int>> vec(5, -1);
	// 示例：向vector添加元素
	vec.push_back(1);
	vec.push_back(2);
	vec.push_back(3);

	// 输出vector中的元素
	for (int i : vec) {
		std::cout << i << std::endl;
	}

	return 0;
}


