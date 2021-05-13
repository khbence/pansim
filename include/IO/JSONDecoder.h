//
//  JSONDecoder.h
//  cytocast
//
//  Created by Áron Takács on 2019. 05. 10..
//

#pragma once
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <customExceptions.h>

#include "rapidjson/rapidjson.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#include "rapidjson/document.h"
#pragma GCC diagnostic pop
#include "rapidjson/error/en.h"

namespace jsond {

    /*
     namespace for the ugly template stuff that is not specific to JSONDecoder
     */
    namespace internal {
        /*
         RecursiveTypelist
         */
        struct Void {};

        template<typename H, typename T>
        struct RecursiveTypelist {};

        /*
         Typelist
         */
        template<typename... T>
        struct Typelist {};

        template<typename... T>
        struct Typelist_Merge {};

        template<typename... T, typename... U>
        struct Typelist_Merge<Typelist<T...>, Typelist<U...>> {
            typedef Typelist<T..., U...> value;
        };

        /*
         Flatten RecursiveTypelist
         */
        template<typename... T>
        struct StripRecursiveList__ {};

        template<typename H, typename... T, typename... U>
        struct StripRecursiveList__<Typelist<RecursiveTypelist<H, RecursiveTypelist<T...>>, U...>> {
            typedef typename StripRecursiveList__<Typelist<RecursiveTypelist<T...>, H, U...>>::value
                value;
        };

        template<typename H, typename... U>
        struct StripRecursiveList__<Typelist<RecursiveTypelist<H, Void>, U...>> {
            typedef Typelist<H, U...> value;
        };

        template<typename... T>
        struct FlattenRecursiveTypelist {};

        template<typename H, typename... T>
        struct FlattenRecursiveTypelist<RecursiveTypelist<H, RecursiveTypelist<T...>>> {
            typedef
                typename StripRecursiveList__<Typelist<RecursiveTypelist<T...>, H>>::value value;
        };

        template<typename H>
        struct FlattenRecursiveTypelist<RecursiveTypelist<H, ::jsond::internal::Void>> {
            typedef ::jsond::internal::Typelist<H> value;
        };

        /*
         Incremental typelist construction

         This solution is based on the genius solution to this problem as
         described here:
         https://stackoverflow.com/questions/24088373/building-a-compile-time-list-incrementally-in-c/24092292#24092292
         */
        template<int N>
        struct Counter : Counter<N - 1> {};
        template<>
        struct Counter<0> {};

#define _START_LIST() \
    static ::jsond::internal::Void __list_maker_helper(::jsond::internal::Counter<__COUNTER__>)

#define _ADD_TO_LIST(type)                                                        \
    static ::jsond::internal::RecursiveTypelist<type,                             \
        decltype(__list_maker_helper(::jsond::internal::Counter<__COUNTER__>{}))> \
        __list_maker_helper(::jsond::internal::Counter<__COUNTER__>)

#define _END_LIST()                                             \
    typedef jsond::internal::FlattenRecursiveTypelist<decltype( \
        __list_maker_helper(::jsond::internal::Counter<__COUNTER__>{}))>::value __member_typelist;
    }// namespace internal

    template<typename Derived>
    struct JSONDecodable;

    /*
     namespace for jsond implementation details
     */
    namespace impl {

        template<typename T>
        struct __is_JSONDecodable {
            const static bool value = std::is_base_of<::jsond::JSONDecodable<T>, T>::value;
        };

        /*
         JSONDecodableMember

         Different wrapper types for object, array, primitive for later
         when decoding to be able to specialize for these.
         */
        template<typename T, int>
        struct JSONDecodableMemberBase {
            typedef T member_type;
            static std::string member_name;
            static T* member_ptr;

            JSONDecodableMemberBase(T* ptr, std::string name) {
                member_name = name;
                member_ptr = ptr;
            }
        };
        template<typename T, int ID>
        std::string JSONDecodableMemberBase<T, ID>::member_name;
        template<typename T, int ID>
        T* JSONDecodableMemberBase<T, ID>::member_ptr;

        template<typename T, int ID>
        struct JSONDecodableMemberObject : public JSONDecodableMemberBase<T, ID> {
            static_assert(__is_JSONDecodable<T>::value,
                "Decodable object must be subclass of JSONDecodable");
            JSONDecodableMemberObject(T* ptr, std::string name)
                : JSONDecodableMemberBase<T, ID>(ptr, name) {}
        };

        template<typename T, typename V, int ID>
        struct JSONDecodableMemberArray : public JSONDecodableMemberBase<T, ID> {
            typedef V contained_type;

            JSONDecodableMemberArray(T* ptr, std::string name)
                : JSONDecodableMemberBase<T, ID>(ptr, name) {}
        };

        template<typename T, int ID>
        struct JSONDecodableMemberPrimitive : public JSONDecodableMemberBase<T, ID> {
            JSONDecodableMemberPrimitive(T* ptr, std::string name)
                : JSONDecodableMemberBase<T, ID>(ptr, name) {}
        };

        /*
         JSONDecodableArrayValue

         Since array values are different in that they do not have their
         own storage or variable name we handle them separately.
         We still have three types object, array, and primitive for sanity.
        */
        template<typename T, int>
        struct JSONDecodableArrayValueBase {
            typedef T member_type;
        };
        template<typename T, int ID>
        struct JSONDecodableArrayValueObject : JSONDecodableArrayValueBase<T, ID> {};
        template<typename T, int ID>
        struct JSONDecodableArrayValuePrimitive : JSONDecodableArrayValueBase<T, ID> {};
        template<typename T, typename V, int ID>
        struct JSONDecodableArrayValueArray : JSONDecodableArrayValueBase<T, ID> {
            typedef V contained_type;
        };

        /*
         __deduce_impl_type

         Used to automatically deduce what the inner type representation for a
         member type should be.
        */
        template<typename T, int ID, bool is_array_member = false>
        struct __deduce_impl_type;

        template<typename T, int ID>
        struct __deduce_impl_type<T, ID, false> {
            typedef typename std::conditional<::jsond::impl::__is_JSONDecodable<T>::value,
                ::jsond::impl::JSONDecodableMemberObject<T, ID>,
                ::jsond::impl::JSONDecodableMemberPrimitive<T, ID>>::type type;
        };

        template<typename T, int ID>
        struct __deduce_impl_type<T, ID, true> {
            typedef typename std::conditional<::jsond::impl::__is_JSONDecodable<T>::value,
                ::jsond::impl::JSONDecodableArrayValueObject<T, ID>,
                ::jsond::impl::JSONDecodableArrayValuePrimitive<T, ID>>::type type;
        };

        template<typename T, typename... Args, template<typename, typename...> class C, int ID>
        struct __deduce_impl_type<C<T, Args...>, ID, false> {
            typedef ::jsond::impl::JSONDecodableMemberArray<C<T, Args...>,
                typename ::jsond::impl::__deduce_impl_type<T, ID, true>::type,
                ID>
                type;
        };

        template<typename T, typename... Args, template<typename, typename...> class C, int ID>
        struct __deduce_impl_type<C<T, Args...>, ID, true> {
            typedef ::jsond::impl::JSONDecodableArrayValueArray<C<T, Args...>,
                typename ::jsond::impl::__deduce_impl_type<T, ID, true>::type,
                ID>
                type;
        };

        // XXX: std::string would be treated as an array so we explicitly
        // specialize it as a primitive
        template<int ID>
        struct __deduce_impl_type<std::string, ID, false> {
            typedef ::jsond::impl::JSONDecodableMemberPrimitive<std::string, ID> type;
        };

        template<int ID>
        struct __deduce_impl_type<std::string, ID, true> {
            typedef ::jsond::impl::JSONDecodableArrayValuePrimitive<std::string, ID> type;
        };

        /*
         Decoding
         */

        // primitive value decoder
        template<typename R>
        struct __primitive_value_decoder {
            static R get(const rapidjson::Value& val) {
                switch (val.GetType()) {
                case rapidjson::Type::kTrueType:
                case rapidjson::Type::kFalseType:
                    return static_cast<R>(val.GetBool());
                case rapidjson::Type::kNumberType:
                    return static_cast<R>(val.GetDouble());
                case rapidjson::Type::kNullType:
                case rapidjson::Type::kStringType:
                case rapidjson::Type::kArrayType:
                case rapidjson::Type::kObjectType:
                    break;
                }

                assert(false);
                return R{};
            }
        };

        template<>
        struct __primitive_value_decoder<std::string> {
            static std::string get(const rapidjson::Value& val) {
                switch (val.GetType()) {
                case rapidjson::Type::kStringType:
                    return val.GetString();
                case rapidjson::Type::kTrueType:
                case rapidjson::Type::kFalseType:
                    return val.GetBool() ? "true" : "false";
                case rapidjson::Type::kNumberType: {
                    std::ostringstream ss;
                    ss << val.GetDouble();
                    return ss.str();
                } break;
                case rapidjson::Type::kNullType:
                case rapidjson::Type::kArrayType:
                case rapidjson::Type::kObjectType:
                    break;
                }

                assert(false);
                return "";
            }
        };

        // object value decoder
        template<typename TL>
        struct __decode_member_list;

        template<typename R>
        struct __object_value_decoder {
            static R get(const rapidjson::Value::Object& obj) {
                R ret;
                __decode_member_list<typename R::__member_typelist>::decode(obj);
                return ret;
            }
        };

        /*
         __array_value_decoder

         Decodes the content type of an array
        */
        template<typename T>
        struct __array_value_decoder;

        template<typename T, int ID>
        struct __array_value_decoder<::jsond::impl::JSONDecodableArrayValuePrimitive<T, ID>> {
            typedef ::jsond::impl::JSONDecodableArrayValuePrimitive<T, ID> primitive_t;

            static T get(rapidjson::Value& val) {
                assert(!val.IsObject() && !val.IsArray());
                return __primitive_value_decoder<typename primitive_t::member_type>::get(val);
            };
        };

        template<typename T, int ID>
        struct __array_value_decoder<::jsond::impl::JSONDecodableArrayValueObject<T, ID>> {
            typedef ::jsond::impl::JSONDecodableArrayValueObject<T, ID> object_t;

            static T get(rapidjson::Value& val) {
                assert(val.IsObject());
                return __object_value_decoder<typename object_t::member_type>::get(val.GetObject());
            }
        };

        template<typename T>
        struct __array_decoder;

        template<typename T, typename V, int ID, template<typename, typename, int> class JSONARRAY>
        struct __array_value_decoder<JSONARRAY<T, V, ID>> {
            typedef JSONARRAY<T, V, ID> array_t;

            static T get(rapidjson::Value& val) {
                assert(val.IsArray());
                return __array_decoder<array_t>::get(val.GetArray());
            }
        };

        /*
        __array_decoder

         Decodes an entire array

         @see JSONARRAY template template type parameter is either:
            * JSONDecodableMemberArray
            * JSONDecodableArrayValueArray
        */
        template<typename T, typename V, int ID, template<typename, typename, int> class JSONARRAY>
        struct __array_decoder<JSONARRAY<T, V, ID>> {
            typedef JSONARRAY<T, V, ID> array_t;

            static T get(const rapidjson::Value::Array& array) {
                T container;

                for (auto it = array.begin(); it != array.end(); ++it) {
                    container.insert(std::end(container),
                        __array_value_decoder<typename array_t::contained_type>::get(*it));
                }

                return container;
            }
        };

        // decoding member with name and storage
        template<typename T>
        struct __decode_member;

        template<typename T, int ID>
        struct __decode_member<::jsond::impl::JSONDecodableMemberPrimitive<T, ID>> {
            static void decode(const rapidjson::Value::Object& obj) {
                typedef ::jsond::impl::JSONDecodableMemberPrimitive<T, ID> member_t;
                auto& tmpMemberName = member_t::member_name;
                auto it = obj.FindMember(tmpMemberName.c_str());
                assert(it != obj.end());
                *member_t::member_ptr =
                    ::jsond::impl::__primitive_value_decoder<typename member_t::member_type>::get(
                        it->value);
            }
        };

        template<typename T, int ID>
        struct __decode_member<::jsond::impl::JSONDecodableMemberObject<T, ID>> {
            static void decode(const rapidjson::Value::Object& obj) {
                typedef ::jsond::impl::JSONDecodableMemberObject<T, ID> member_t;
                std::string& tmp = member_t::member_name;
                auto it = obj.FindMember(tmp.c_str());
                assert(it != obj.end());
                assert(it->value.IsObject());
                *member_t::member_ptr =
                    ::jsond::impl::__object_value_decoder<T>::get(it->value.GetObject());
            }
        };

        template<typename T, typename V, int ID>
        struct __decode_member<::jsond::impl::JSONDecodableMemberArray<T, V, ID>> {
            static void decode(const rapidjson::Value::Object& obj) {
                typedef ::jsond::impl::JSONDecodableMemberArray<T, V, ID> member_t;
                std::string& tmp = member_t::member_name;
                auto it = obj.FindMember(tmp.c_str());
                // auto it = obj.FindMember(member_t::member_name.c_str());
                assert(it != obj.end());
                assert(it->value.IsArray());
                *member_t::member_ptr = __array_decoder<member_t>::get(it->value.GetArray());
            }
        };

        // decoding members of and object
        template<typename H, typename... T>
        struct __decode_member_list<::jsond::internal::Typelist<H, T...>> {
            static void decode(const rapidjson::Value::Object& obj) {
                __decode_member<H>::decode(obj);
                __decode_member_list<::jsond::internal::Typelist<T...>>::decode(obj);
            }
        };

        template<typename H>
        struct __decode_member_list<::jsond::internal::Typelist<H>> {
            static void decode(const rapidjson::Value::Object& obj) {
                __decode_member<H>::decode(obj);
            }
        };


    }// namespace impl

/*
 JSON declaration macros
 */
#define BEGIN_MEMBER_DECLARATIONS() _START_LIST()

#define DECODABLE_MEMBER(arg0_type, arg1_name)                                       \
    arg0_type arg1_name;                                                             \
    typedef typename ::jsond::impl::__deduce_impl_type<arg0_type, __COUNTER__>::type \
        __##arg1_name##_decodable_member_type;                                       \
    __##arg1_name##_decodable_member_type __##arg1_name##_decodable_member =         \
        __##arg1_name##_decodable_member_type(&arg1_name, #arg1_name);               \
    _ADD_TO_LIST(__##arg1_name##_decodable_member_type)

#define END_MEMBER_DECLARATIONS() _END_LIST()

    /*
     JSONDecodable
     */
    template<typename Derived>
    struct JSONDecodable {
        static Derived Decode(std::string&& json_str) {
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(json_str.c_str());
            if (!ok) {
                std::cerr << "JSON parse error: " << rapidjson::GetParseError_En(ok.Code())
                          << std::count(json_str.begin(), json_str.begin() + ok.Offset(), '\n')
                          << '\n';
                exit(EXIT_FAILURE);
            }

            // only support json that has an object as root
            assert(d.IsObject());

            return ::jsond::impl::__object_value_decoder<Derived>::get(d.GetObject());
        }

        static Derived DecodeFromFile(const std::string& fileName) {
            std::ifstream t(fileName.c_str());
            if (!t.is_open()) { throw init::BadInputFile(fileName); }
            std::string str;

            t.seekg(0, std::ios::end);
            str.reserve(t.tellg());
            t.seekg(0, std::ios::beg);

            str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
            return Decode(std::move(str));
        }

#define DECODE_JSON_FILE(fileName, formatClass) \
    jsond::JSONDecodable<formatClass>::DecodeFromFile(fileName)
    };
}// namespace jsond
