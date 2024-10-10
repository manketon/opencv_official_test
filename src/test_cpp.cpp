#include "test_cpp.h"
#include <iostream>
#include <vector>

// �Զ��������
template<class _Ty>
class CustomAllocator : public std::allocator<_Ty> {
public:
	template<class U>
	struct rebind {
		typedef CustomAllocator<U> other;
	};

	CustomAllocator() throw() {}

	CustomAllocator(const CustomAllocator& other) throw()
		: std::allocator<_Ty>(other) {}

	template<class U>
	CustomAllocator(const CustomAllocator<U>& other) throw()
		: std::allocator<_Ty>(other) {}

	// ��������������Զ����ڴ�����߼�

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
#if _HAS_DEPRECATED_ALLOCATOR_MEMBERS
	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD __declspec(allocator) _Ty* allocate(
		_CRT_GUARDOVERFLOW const size_t _Count, const void*) 
	{
		std::cout << __FUNCTION__ << " | line:" << __LINE__ << std::endl;
		return allocate(_Count);
	}

	template <class _Objty, class... _Types>
	_CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS void construct(_Objty* const _Ptr, _Types&&... _Args) {
		std::cout << __FUNCTION__ << " | line:" << __LINE__ << std::endl;
		::new (std::_Voidify_iter(_Ptr)) _Objty(_STD forward<_Types>(_Args)...); //���Ѿ�����õ��ڴ��Ϲ�����󣨲��·����ڴ棩
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
	// ʹ���Զ����������ʼ��vector
	std::vector<int, CustomAllocator<int>> vec(5, -1);
	// ʾ������vector���Ԫ��
// 	vec.push_back(1);
// 	vec.push_back(2);
// 	vec.push_back(3);

	// ���vector�е�Ԫ��
	for (int i : vec) {
		std::cout << i << std::endl;
	}

	return 0;
}


