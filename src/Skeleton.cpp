//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Adam Zsombor
// Neptun : X079FB
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f; 
	T d;  
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 20;

struct Camera { 
	vec3 wEye, wLookat, wVup;   
	float fov, asp, fp, bp;
    
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 45.0f * M_PI / 180.0f;
		fp = 0.5f;
        bp = 20;
	}
    
	mat4 V() { 
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { 
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
    
public:
	Material(vec3 _kd, vec3 _ks, vec3 _ka, int s) {
		kd = _kd;
		ks = _ks;
		ka = _ka;
		shininess = s;
	}
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
};

struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[8] lights;
		uniform int   nLights;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight[8];
		in  vec2 texcoord;
		
        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
    
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

class Geometry {
protected:
	unsigned int vao, vbo;
    
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}

	void Load(const std::vector<VertexData>& vtxData) {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

    virtual void Draw() {}

	virtual void Move(float d) {}
	
    void addTriangle(std::vector<VertexData>& vtxData, vec3 a, vec3 b, vec3 c) {
        VertexData v;
        v.normal = cross(c - a, b - a);
        v.texcoord = vec2(0, 0);
        v.position = a;
        vtxData.push_back(v);
        v.position = b;
        vtxData.push_back(v);
        v.position = c;
        vtxData.push_back(v);
    }

	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
protected:
	unsigned int nVtxPerStrip, nStrips;
    
public:
	ParamSurface() {
        nVtxPerStrip = 0;
        nStrips = 0;
    }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	
		for (int i = 0; i < N; i++) { 
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		Load(vtxData);
	}

	void Draw() {
		glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) {
            glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
        }
	}
};

class Pyramid : public Geometry {
	std::vector<VertexData> vtxData;
    
public:
	Pyramid() {
		build();
	}

	void build() {
		vec3 A = vec3(1, 0.75f, 0);
		vec3 B = vec3(-1, 0.75f, 0);
		vec3 C = vec3(-1, -0.75f, 0);
		vec3 D = vec3(1, -0.75f, 0);
		vec3 E = vec3(0, 0, 0.8f);
		addTriangle(vtxData, A, B, C);
		addTriangle(vtxData, A, C, D);
		addTriangle(vtxData, A, B, E);
		addTriangle(vtxData, B, C, E);
		addTriangle(vtxData, C, D, E);
		addTriangle(vtxData, D, A, E);
		Load(vtxData);
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 18);
	}
};

class Circle : public Geometry {
public:
	Circle() {
		build();
	}
    
	void build() {
		std::vector<VertexData> vtxData;
		for (int i = 0; i < tessellationLevel; i++) {
			float p = 2.0f * M_PI * i / tessellationLevel;
			vtxData.push_back({ vec3(0.5f * sinf(p), 0.5f * cosf(p), 0.8f), vec3(0.5f * sinf(p), 0.5f * cosf(p), 1), vec2(0.5f * sinf(p), 0.5f * cosf(p)) });
		}
		Load(vtxData);
	}
    
	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 20);
	}
	
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {}
};

class BigCylinder : public ParamSurface{
public:
	BigCylinder() {
		create();
	}
    
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        U = U * 2.0f * M_PI;
        V = V;
		X = Cos(U);
        Y = Sin(U);
        Z = V;
	}
};

class LongCylinder : public ParamSurface{
public:
    LongCylinder() {
        create();
    }
    
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        U = U * 2.0f * M_PI;
        V = V;
        X = V * 1.5f;
        Y = Sin(U) * 0.1f;
        Z = Cos(U) * 0.1f;
    }
};

float line = 6 * 0.3f;
float curve = 0.3f * M_PI;
float vr = 0, vl = 0;

class Bit : public Geometry {
    std::vector<VertexData> vtxData;
    vec3 startPoint;
    float alpha, start, r, pi_2;
    
public:
    Bit(float s, float y) : start(s), startPoint(vec3(-0.9f, y, -0.3f)), r(0.3f), pi_2(M_PI / 2) {
        create();
    }
    
    void create() {
        createSurface(updatePosition());
        Load(vtxData);
    }

    vec3 updatePosition() {
        while (start < 0) {
            start += 2 * (line + curve);
        }
        
        while (start >= 2 * (line + curve)) {
            start -= 2 * (line + curve);
        }
        
        vec3 p;

        if (start >= 0 && start < line) {
            p = vec3(startPoint.x + start, startPoint.y, startPoint.z);
            alpha = -M_PI;
        } else if (start >= line && start < line + curve) {
            p.x = startPoint.x + 6 * r + cosf((start - line) / curve * M_PI - pi_2) * r;
            p.y = startPoint.y;
            p.z = startPoint.z + r + sinf((start - line) / curve * M_PI - pi_2) * r;
            alpha = ((start - line) / curve - 1) * M_PI;
        } else if (start >= line + curve && start < 2 * line + curve) {
            p.x = startPoint.x + 6 * r - (start - line - curve);
            p.y = startPoint.y;
            p.z = startPoint.z + 2 * r;
            alpha = 0;
        } else {
            p.x = startPoint.x + cosf((start - 2 * line - curve) / curve * M_PI + pi_2) * r;
            p.y = startPoint.y;
            p.z = startPoint.z + r + sinf((start - 2 * line - curve) / curve * M_PI + pi_2) * r;
            alpha = (start - 2 * line - curve) / curve * M_PI;
        }
        
        return p;
    }

    void createSurface(vec3 p0) {
        vec3 a, b, c, d;
        if (alpha != 0) {
            a = vec3(p0.x - (0.05f) * cosf(alpha), p0.y - (0.2f), p0.z - sinf(alpha) * (0.05f));
            b = vec3(p0.x + (0.05f) * cosf(alpha), p0.y - (0.2f), p0.z + sinf(alpha) * (0.05f));
            c = vec3(p0.x + (0.05f) * cosf(alpha), p0.y + (0.2f), p0.z + sinf(alpha) * (0.05f));
            d = vec3(p0.x - (0.05f) * cosf(alpha), p0.y + (0.2f), p0.z - sinf(alpha) * (0.05f));
        }
        else {
            a = vec3(p0.x - (0.05f), p0.y - (0.2f), p0.z);
            b = vec3(p0.x + (0.05f), p0.y - (0.2f), p0.z);
            c = vec3(p0.x + (0.05f), p0.y + (0.2f), p0.z);
            d = vec3(p0.x - (0.05f), p0.y + (0.2f), p0.z);
        }
        addTriangle(vtxData, a, b, c);
        addTriangle(vtxData, a, c, d);
    }

    std::vector<VertexData> getVtxData() {
        return vtxData;
    }
};

class Feet : public Geometry {
    std::vector<Bit*> bits;
    std::vector<VertexData> vtxData;
    float side;
    
public:
    Feet(float s) : side(s) {
        create();
    }
    
    void moveBits(float delta) {
        float start = 0;
        while (start < 2 * (line + curve)) {
            Bit* bit = new Bit(start - delta, side);
            start += 0.15f;
            bits.push_back(bit);
            for (int i = 0; i < bit->getVtxData().size(); i++) {
                vtxData.push_back(bit->getVtxData()[i]);
            }
        }
    }

    void create() {
        moveBits(0);
        Load(vtxData);
    }

    void Move(float deltaD) {
        for (int i = 0; i < bits.size(); i++) {
            delete bits[i];
        }
        bits.clear();

        vtxData.clear();
        moveBits(deltaD);
        Load(vtxData);
    }

    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 228);
    }
};

class Plane : public ParamSurface {
    float x, y;
    
public:
		Plane() : x(5), y(5) {
			create();
		}
    
		void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
            U = U;
            V = V;
			X = U * x - x / 2;
            Y = V * y - y / 2;
            Z = 0;
		}
};

struct Object {
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;

	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0),
        shader(_shader), texture(_texture), material(_material), geometry(_geometry) {}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void animate(float tstart, float tend) {}
};

Material* tankMat;
Material* planeMat;
Texture* tankTex;
Texture* planeTex;

class Tank : public Object {
	vec3 point;
	float alpha;
    
public:
	Tank() : Object(new PhongShader(), tankMat, tankTex, new Pyramid()),
        point(vec3(0, 0, 0)), alpha(0) {}
    
    void animate(float tstart, float tend) {
        if (vl != 0 && vr != 0) {
            vec3 direction = vec3(cosf(alpha), sinf(alpha), 0);
            vec3 newP = point + vec3(direction * ((vl + vr) / 2) * (tend - tstart));
            point.x = newP.x;
            point.y = newP.y;
            point.z = newP.z;
        }
        
        alpha = alpha + ((vr - vl) / 1.5f) * (tend - tstart);
        rotationAngle = alpha;
        translation = point;
    }
    
    vec3 getPoint() {
        return point;
    }
};

Tank* tank;

mat4 tankMove() {
	return TranslateMatrix(tank->translation);
}

mat4 tankMoveBack() {
	return TranslateMatrix(-tank->translation);
}

mat4 tankRotation() {
	return RotationMatrix(tank->rotationAngle, vec3(0, 0, 1));
}

mat4 tankRotationBack() {
	return RotationMatrix(-tank->rotationAngle, vec3(0, 0, 1));
}

class Ground : public Object {
public:
	Ground() : Object(new PhongShader(), planeMat, planeTex, new Plane()) {
		scale = vec3(10000, 10000, 1);
		translation = vec3(0, 0, -0.3f);
	}
};

class Tower : public Object {
public:
	Tower() : Object(new PhongShader(), tankMat, tankTex, new BigCylinder()) {
        scale = vec3(0.5, 0.5, 0.8);
    }
    
	void spin(float angle, float speed) {
		rotationAngle = angle;
		rotationAxis = vec3(0, 0, 1);
	}

	void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * tankRotation() * TranslateMatrix(translation)  * tankMove();
		Minv = tankMoveBack() * tankRotationBack() * TranslateMatrix(-translation)  * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
};

Tower* tower;

mat4 towerRotation() {
	return RotationMatrix(tower->rotationAngle, vec3(0, 0, 1));
}

class Cannon : public Object {
public:
	Cannon() : Object(new PhongShader(), tankMat, tankTex, new LongCylinder()) {
		translation = vec3(-0.3f, 0, 0.6f);
        rotationAxis = vec3(0, 1, 0);
	}

	void lift(float angle) {
		rotationAngle = angle;
	}

	void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * towerRotation() * tankRotation() * TranslateMatrix(translation) * tankMove();
		Minv = tankMoveBack() * TranslateMatrix(-translation) * tankRotationBack() * towerRotation() * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
};

class RightChain : public Object {
    float delta;
    
public:
	RightChain() : Object(new PhongShader(), tankMat, tankTex, new Feet(-0.75f)), delta(0) {}
    
	void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation) * tankRotation() * tankMove();
		Minv = tankMoveBack() * TranslateMatrix(-translation) * tankRotationBack() * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
    
    void animate(float tstart, float tend) {
        delta += vr * (tend - tstart);
        geometry->Move(delta);
    }
};

class LeftChain : public Object {
    float delta;
    
public:
	LeftChain() : Object(new PhongShader(), tankMat, tankTex, new Feet(0.75f)) {}
    
	void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation) * tankRotation() * tankMove();
		Minv = tankMoveBack() * tankRotationBack() * TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
    
    void animate(float tstart, float tend) {
        delta += vl * (tend - tstart);
        geometry->Move(delta);
    }
};

class Top : public Object {
public:
	Top() : Object(new PhongShader(), tankMat, tankTex, new Circle()) {}
    
	void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation) * tankRotation() * tankMove();
		Minv = tankMoveBack() * tankRotationBack() * TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
};

class RandomObject : public Object {
public:
    RandomObject(int a, int b) : Object(new PhongShader(), planeMat, planeTex, new Pyramid()) {
        translation = vec3(a, b, -0.3f);
        scale = vec3(1, 1, 6);
    }
};

Camera camera;
std::vector<Object*> objects;
std::vector<Light> lights;
Cannon* cannon;
Top* top;
RightChain* right;
LeftChain* left;

void buildTank() {
    left = new LeftChain();
    cannon = new Cannon();
    top = new Top();
    right = new RightChain();
    tank = new Tank();
    tower = new Tower();

    objects.push_back(tank);
    objects.push_back(tower);
    objects.push_back(cannon);
    objects.push_back(top);
    objects.push_back(left);
    objects.push_back(right);
}

void buildGround() {
	objects.push_back(new Ground());
    objects.push_back(new RandomObject(7, 4));
    objects.push_back(new RandomObject(4, 7));
    objects.push_back(new RandomObject(9, 6));
    objects.push_back(new RandomObject(6, 9));
    objects.push_back(new RandomObject(7, -4));
    objects.push_back(new RandomObject(4, -7));
    objects.push_back(new RandomObject(9, -6));
    objects.push_back(new RandomObject(6, -9));
}

void setCamera() {
    camera.wEye = vec3(tank->getPoint().x - 5, tank->getPoint().y, tank->getPoint().z + 1.5f);
    camera.wLookat = vec3(tank->getPoint().x, tank->getPoint().y, tank->getPoint().z + 0.8f);
    camera.wVup = vec3(0, 0, 1);
}

void conf() {
    setCamera();
	lights.resize(1);
	lights[0].wLightPos = vec4(1, 2, 3, 0);
	lights[0].La = vec3(0.2f, 0.2f, 0.2f);
	lights[0].Le = vec3(1, 1, 1);
}

void Render() {
	RenderState state;
	state.wEye = camera.wEye;
	state.V = camera.V();
	state.P = camera.P();
	state.lights = lights;
    for (Object* obj : objects){
        obj->Draw(state);
    }
}

void Animate(float start, float end) {
    if (vl != 0) {
        left->animate(start, end);
    }
    if (vr != 0) {
        right->animate(start, end);
    }
    if (vr != 0 || vl != 0) {
        tank->animate(start, end);
    }
}

class PlaneTexture : public Texture {
public:
    PlaneTexture(const int width, const int height) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 red = vec4(0.863f, 0.078f, 0.235f, 1.0f);
        const vec4 blue = vec4(0.255f, 0.412f, 0.882f, 1.0f);
        const vec4 green = vec4(0.133f, 0.545f, 0.133f, 1.0f);
        const vec4 yellow = vec4(1.0f, 0.875f, 0.0f, 1.0f);
        srand(time(0));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int randomValue = rand() % 100;
                if (randomValue < 25) {
                    image[y * width + x] = red;
                } else if (randomValue < 50) {
                    image[y * width + x] = blue;
                } else if (randomValue < 75) {
                    image[y * width + x] = green;
                } else {
                    image[y * width + x] = yellow;
                }
            }
        }
        create(width, height, image, GL_NEAREST);
    }
};

void createMaterials() {
    tankMat = new Material(vec3(0.4f, 0.4f, 0.4f), vec3(1, 1, 1), vec3(0.2f, 0.2f, 0.2f), 5);
    planeMat = new Material(vec3(1, 1, 1), vec3(2, 2, 2), vec3(1, 1, 1), 20);
}

void createTextures() {
    tankTex = new PlaneTexture(0, 0);
    planeTex = new PlaneTexture(1024, 2048);
}

void onInitialization() {
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
    createMaterials();
    createTextures();
    buildTank();
	buildGround();
	conf();
}

void onDisplay() {
	glClearColor(0.3f, 0.4f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, windowWidth, windowHeight);
	Render();
    setCamera();

	glutSwapBuffers();
}

float angle = 0;
float pipeAngle = 0;

void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
        case 'o':
            vr += 0.1f;
            break;
        case 'l':
            vr -= 0.1f;
            break;
        case 'q':
            vl += 0.1f;
            break;
        case 'a':
            vl -= 0.1f;
            break;
        case 'b':
            angle += 15 * M_PI / 180;
            tower->spin(angle, 0);
            break;
        case 'n':
            angle -= 15 * M_PI / 180;
            tower->spin(angle, 0);
            break;
        case 'w':
            pipeAngle -= 5 * M_PI / 180;
            cannon->lift(pipeAngle);
            break;
        case 's':
            pipeAngle += 5 * M_PI / 180;
            if (pipeAngle < 75 * M_PI / 180 && pipeAngle > -45 * M_PI / 180) {
                cannon->lift(pipeAngle);
            }
            break;
    }
    glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onIdle() {
    static float tend = 0;
    const float dt = 0.1f;
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        Animate(t, t + Dt);
    }
    glutPostRedisplay();
}
