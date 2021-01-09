# SPA类前后端完全分类应用使用Authing的云身份验证与单点登录

## 为什么需要云身份验证和单点登录

简单来说是为了降低维护用户注册登录系统、权限、统计等各方面的成本。

## 应用结构简述

通过Authing实现身份验证和单点登录，有很多种方法，这篇文章的例子是根据自身软件架构实现了其中一种相对简单的方法，并不适用所有情况，Authing本身还提供了多种的登录解决方案，包括直接嵌入到网站上、APP上的等等。

前端采用纯 React/React-router/Ant.design 开发，没用 Redux/Server Rendering 之类比较复杂的东西，就使用 create-react-app 的最基本方案，没用TypeScript（因为懒，我有罪）。

后端采用Python + FastAPI的简单API。

## 登录流程

### 第一阶段，前端

通过检测本地localStorage，未发现保存的登录token信息时，提示用户需要登录，给出登录链接，用HTML的a标签直接跳转到authing提供的SSO网址上，例如 xxxx.authing.cn ，其中xxxx是可以用户自定义的。

### 第二阶段，Authing SSO 网站

完成登录，可以自由配置，例如注册方式，登录方式比如游记验证码，微信小程序，微信扫码，邮箱密码等。登录成功后，会自动跳转到你配置的回调地址上，回调时可以选择直接提供token。

例如你配置的回调地址是  xxxx.cn/login  ，authing可以通过配置，在登录成功后自动跳转到 xxxx.cn/login/#/token=xxxxxxxx

### 第三阶段，通过回调返回前端

这样就可以直接在前端，即React部分通过对window.location或document.URL的解析获取到这个token。

前端获取到这个token，就可以通过authing提供的JavaScript的SDK，验证token，获取用户信息。如果获取用户信息成功，则说明用户登录成功。

如果在第一阶段中，通过localStorage检测到了本地的token，可以直接跳转到这一阶段通过authing的SDK进行token验证，这样就跳过了第二阶段。

前端对后端的每个API调用都要提交token，可以通过设置header的方式实现。

### 第四阶段，后端

API拿到前端的token之后，通过authing提供的python SDK，验证这个token和获取用户当前信息，通过后端再次验证这个token是否合法，如果不合法可以返回401未授权登录，如果合法，可以继续实现API本身的功能。

## 用户的体验流程

* 未登录时：
	* 用户打开网站，前端提示未登录，用户点击登录链接（或按钮），跳转到Authing的SSO网址
	* 用户在Authing网站上实现统一的注册/登录，成功后跳转回网站
	* 跳转回的回调地址通过Token可以验证用户登录成功，所以这里用户可以直接看到登录成功的提示
	* 用户开始使用应用
* 登录后时：
	* 用户打开网站，因为前端已经检测到了保存的token，并且通过sdk验证了前端token的基本有效性（实际有效性是又后端验证的），所以直接跳转到应用部分
	* 用户开始使用应用

## 开发的体验

* 前端：
	* 使用Authing-js-sdk验证token
	* 使用Authing-sso-sdk实现彻底退出sso登录
* 后端：
	* 使用authing-python SDK验证前端传过来的token
* 其他：
	* 理论上用户可以通过伪造token，骗过前端程序，但是因为后端每次API调用都会验证token，后端的token合法性验证是对前端透明的，所以无法欺骗后端程序，只要后端验证不通过，就可以不给前端返回机密信息。
	* 根据是否允许用户在多个地方登录（如多个电脑、浏览器登录），可以有两种策略，一种是允许用户多个地方登录，那不需要做太多测试；另一种是只允许用户在最后登录的设备中使用，这个时候可以通过对比从authing sdk拿到的token（最后登录的token）和用户前端提交的token来实现这个能力。

## Authing实现的云身份验证和SSO的优点

* 不用实现与维护自己的用户信息系统，包括用户注册、登录、找回密码等
* 可以快速实现多种登录方式，如邮箱登录、手机验证码登录、微信扫码登录等
* 可以通过Authing直接实现用户权限控制功能，通过用户分组等等方法
* 用户登录信息保存在Authing的单点登录（SSO）上，只要登录信息没过期，就可以让用户继续快速登录，提高用户体验，而这些都可以通过配置实现
* 直接对接Authing的用户统计功能，包括活跃用户，登录日志等等，不需要额外的实现

## 代码

代码分为前端和后端两部分

### 前端

前端分为四个主要部分：
* 检测登录状态，未登录时跳转到Authing SSO的组件
* 接收Authing回调信息的landing页面，完成登录token验证的组件
* 退出登录功能
* 封装浏览器的AJAX接口，在提交时携带token


跳转到Authing SSO

```javascript

/**
 * 本地先检测登录状态，如果没有则提示跳转到authing sso登录
 */
export function checkLogin() {
    const userInfo = localStorage.getItem('userInfo')
    if (userInfo) {
        try {
            return JSON.parse(userInfo)
        } catch (e) {
            console.error(e)
        }
    }
}


<a
	href='https://xxx.authing.cn'
	alt='login'
	style={{
		color: 'black',
	}}
>
	<Button type="primary" key="console">
		请点击这里，在新页面完成登录
	</Button>
</a>

```

登录成功后，authing调用设置的回调地址，在跳转过来的landing页面中，可以通过URL拿到token

```javascript

import { AuthenticationClient } from "authing-js-sdk"

const authenticationClient = new AuthenticationClient({
    appId: "XXXXXXXXXXXXXX",
})

// 从URL获取token
const m = window.location.hash.match(/id_token=([^$&]+)[$&]/)
if (m) {
	const token = m[1]
	// 设置好客户端token之后获取用户信息
	authenticationClient.setToken(token)
	const userInfo = await authenticationClient.getCurrentUser()
	if (!userInfo || !userInfo.id) {
		this.setState({
			loading: false,
			loginSuccess: false,
			text: '登录失败'
		})
		return
	}
	// 成功
	localStorage.setItem('userInfo', JSON.stringify(userInfo))
	this.setState({
		loading: false,
		loginSuccess: true,
		text: '登录成功'
	})
}
```

退出登录

```javascript

import AuthingSSO from "@authing/sso"

export const authSSO = new AuthingSSO({
    appId: "XXXXXXXXXXXXXX",
    appType: "oidc",
    appDomain: "xxx.authing.cn"
})


<Button
	type='link'
	onClick={async ()=> {
		try {
			// 调用接口，从SSO页面也退出登录
			console.log(await authSSO.logout())
		} catch(e) { }
		localStorage.removeItem('userInfo')
		message.warn('您已经退出登录')
	}}
>
	{checkLogin() ? checkLogin().nickname + ' 退出登录' : ''}
</Button>
```

对API提交时，同时携带token，以便于后端验证用户权限

```javascript

/**
 * 这个函数是用来代替原生的fetch函数
 */
export async function fetchGet(url) {
    const userInfo = checkLogin()
	const token = userInfo ? userInfo.token : ''
    const res = await fetch(url, {
        headers: {
            'token': token
        }
    })
	// 后端授权检测失败
    if (res.status === 401) {
        message.error('您已经退出登录')
        localStorage.removeItem('userInfo')
        return res
    }
    return res
}

```

### 后端

后端主要是接收前端传过来的token并验证，这个组件对于不同框架来说可以是一个middleware，也可以根据需要设计成一个装饰器。

以下代码针对FastAPI设计

```python
from fastapi import FastAPI

# https://github.com/tiangolo/fastapi/issues/142#issuecomment-688566673
from fastapi import Security
from fastapi.security.api_key import APIKeyHeader
from fastapi import HTTPException
from starlette import status

from authing.v2.authentication import AuthenticationClient, AuthenticationClientOptions

authentication_client = AuthenticationClient(
    options=AuthenticationClientOptions(
        user_pool_id='XXXXXXXXXXXXXX'
))


def auth_testing(token):
    """
    测试authing的结果
    """
	user = authentication_client.get_current_user(token)
	user_id = user.get('id')
	user_token = user.get('token')
	if token == user_token:
		return True, '成功'
    return False, '验证失败'


# 设置FastAPI要获取的header名称
API_KEY_NAME = "token"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key_header: str = Security(api_key_header_auth)):
    ret, detail = auth_testing(api_key_header)
    if not ret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        )


# 用法：
# @app.get('/', dependencies=get_dependencies())
def get_dependencies():
    return [Security(get_api_key)]
        

app = FastAPI()


@app.get('/', dependencies=get_dependencies())
def main():
    return { 'ok': 'hello' }

```
