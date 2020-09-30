#include <vector>
#include <cmath>
#include <iostream> 
#include <limits>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);
const TGAColor green = TGAColor(0, 255, 0, 255);
const TGAColor blue = TGAColor(0, 0, 255, 255);

Model* model = NULL;
int* zbuffer = NULL;
const int width = 800;
const int height = 800;
const int depth = 255;

Vec3f light_dir(0, 0, -1);
Vec3f camera(0, 0, 3);

Vec3f m2v(Matrix m)
{
	// x y z 除以 w
	return Vec3f(m[0][0] / m[3][0], m[1][0] / m[3][0], m[2][0] / m[3][0]);
}


// 将(x,y,z) 变成矩阵 [x, y ,z ,1] 
Matrix v2m(Vec3f v)
{
	Matrix m(4, 1);
	m[0][0] = v.x;
	m[1][0] = v.y;
	m[2][0] = v.z;
	m[3][0] = 1.0f;
	return m;
}

Matrix viewport(int x, int y, int w, int h)
{
	// cube [-1,1]*[-1,1]*[-1,1] is mapped onto the screen cube [x,x+w]*[y,y+h]*[0,d].
	Matrix m = Matrix::identity(4);
	m[0][3] = x + w / 2.0f;
	m[1][3] = y + w / 2.0f;
	m[2][3] = depth + w / 2.0f;

	m[0][0] = w / 2.0f;
	m[1][1] = h / 2.0f;
	m[2][2] = depth / 2.0f;
	return m;
}

void line(int x0, int y0, int x1, int y1, TGAImage& image, TGAColor color)
{
	bool steep = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1)) // 如果线条陡峭，我们要转置图像(这么做的原因是，我们是用x循环，如果斜率大于1，那么y方向上的每个点的差值大于1，所以会有离散的点出现)
											   // 转置了之后，画(y,x) 这样就相当于用y循环，斜率小于1，x不会离散
	{
		std::swap(x0, y0);
		std::swap(x1, y1);
		steep = true;
	}
	if (x0 > x1) // 让它从左到右
	{
		std::swap(x0, x1);
		std::swap(y0, y1);
	}
	int dx = x1 - x0;
	int dy = y1 - y0;
	//float derror = std::abs(dy / float(dx)); // 斜率
	//float error = 0;
	int derror2 = std::abs(dy) * 2; // 为了摒弃除法，提高效率，derror2 = derror * dx,为了去掉0.5 再放大2倍
	int error2 = 0;
	int y = y0;
	for (int x = x0; x <= x1; x++)
	{
		if (steep)
		{
			image.set(y, x, color); // 如果转置过了，反转置回来
		}
		else
		{
			image.set(x, y, color);
		}
		//error += derror;   // 斜率叠加到error上，如果大于0.5了，说明这时候y要朝着斜率移动了，1或者-1个单位，然后既然生效了，error就要减去移动的这个单位1，然后继续叠加
						   // （可以想象这样画出来的直线是锯齿状的偏移）
		/*if (error > 0.5)
		{
			y += (y1 > y0 ? 1 : -1);
			error -= 1.0;
		}*/
		error2 += derror2;
		if (error2 > dx)
		{
			y += (y1 > y0 ? 1 : -1);
			error2 -= dx * 2;
		}

	}
}


void line(Vec2i t0, Vec2i t1, TGAImage& image, TGAColor color)
{
	line(t0.x, t0.y, t1.x, t1.y, image, color);
}
//
//Vec3f barycentric(Vec2i* pts, Vec2i P)
//{
//	// 做(ABx, ACx, PAx) 和 (ABy, ACy, PAy) 的叉积
//	Vec3f u = cross(Vec3f(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - P.x),
//		Vec3f(pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - P.y));
//	// pts 和 p 都是int类型的，所以如果 u.z 绝对值小于1，那么 u.z = 0
//	// u(u, v, 1)的z为0，那么三角形退化了 
//	if (std::abs(u.z) < 1) return Vec3f(-1, 1, 1);
//	return Vec3f(1.0f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
//}
//
//Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P)
//{
//	Vec3f s[2];
//	for (int i = 2; i--; )
//	{
//		s[i][0] = C[i] - A[i];
//		s[i][1] = B[i] - A[i];
//		s[i][2] = A[i] - P[i];
//	}
//	Vec3f u = cross(s[0], s[1]);
//	if (std::abs(u[2]) > 1e-2)
//		return Vec3f(1.0f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
//	return Vec3f(-1, 1, 1);
//}

//void triangle(Vec2i* pts, TGAImage& image, TGAColor color)
//{
//	Vec2i bboxmin(image.get_width() - 1, image.get_height() - 1);
//	Vec2i bboxmax(0, 0);
//	Vec2i clamp(image.get_width() - 1, image.get_height() - 1);
//	for (int i = 0; i < 3; ++i)
//		for (int j = 0; j < 2; ++j)
//		{
//			bboxmin[j] = std::fmax(0, std::fmin(bboxmin[j], pts[i][j]));
//			bboxmax[j] = std::fmax(clamp[j], std::fmax(bboxmax[j], pts[i][j]));
//		}
//	Vec2i P;
//	// 遍历包围盒
//	for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++)
//	{
//		for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++)
//		{
//			Vec3f bc_screen = barycentric(pts, P);
//			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
//			image.set(P.x, P.y, color);
//		}
//	}
//}

void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage& image, TGAColor color)
{
	if (t0.y == t1.y && t0.y == t2.y) return; //  三角形退化成一条线了
	// sort by y
	if (t0.y > t1.y) std::swap(t0, t1);
	if (t0.y > t2.y) std::swap(t0, t2);
	if (t1.y > t2.y) std::swap(t1, t2);
	/*line(t0, t1, image, green);
	line(t1, t2, image, green);
	line(t2, t0, image, red);*/
	int total_height = t2.y - t0.y;
	//for (int y = t0.y; y <= t1.y; ++y)  // 下半部分
	//{
	//	int segment_height = t1.y - t0.y + 1;
	//	float alpha = (float)(y - t0.y) / total_height;
	//	float beta = (float)(y - t0.y) / segment_height;
	//	Vec2i A = t0 + (t2 - t0) * alpha;
	//	Vec2i B = t0 + (t1 - t0) * beta;
	//	//image.set(A.x, y, red);
	//	//image.set(B.x, y, green);
	//	if (A.x > B.x) std::swap(A, B);
	//	for (int j = A.x; j <= B.x; ++j)
	//	{
	//		image.set(j, y, color);
	//	}
	//}
	//for (int y = t1.y; y <= t2.y; ++y) // 上半部分
	//{
	//	int segment_height = t2.y - t1.y + 1;
	//	float alpha = (float)(y - t0.y) / total_height;
	//	float beta = (float)(y - t1.y) / segment_height;
	//	Vec2i A = t0 + (t2 - t0) * alpha;
	//	Vec2i B = t1 + (t2 - t1) * beta;
	//	//image.set(A.x, y, red);
	//	//image.set(B.x, y, green);
	//	if (A.x > B.x) std::swap(A, B);
	//	for (int j = A.x; j <= B.x; ++j)
	//	{
	//		image.set(j, y, color);
	//	}
	//}
	for (int i = 0; i < total_height; ++i)
	{
		bool second_half = i > t1.y - t0.y || t1.y == t0.y; // 为true的时候是上半部分，渲染顺序依然是先下半部分，后上半部分
		int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
		float alpha = (float)i / total_height;
		float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height;
		Vec2i A = t0 + (t2 - t0) * alpha;
		Vec2i B = second_half ? t1 + (t2 - t1) * beta : t0 + (t1 - t0) * beta;
		if (A.x > B.x) std::swap(A, B);
		for (int j = A.x; j <= B.x; ++j)
		{
			image.set(j, t0.y + i, color);
		}

	}
}

// 给出一段伪代码，拿到每个三角形的包围盒，然后去枚举里面的点，如果在三角形里面，那么着色
//triangle(vec2 points[3]) {
//	vec2 bbox[2] = find_bounding_box(points);
//	for (each pixel in the bounding box) {
//		if (inside(points, pixel)) {
//			put_pixel(pixel);
//		}
//	}
//}

void rasterize(Vec2i p0, Vec2i p1, TGAImage& image, TGAColor color, int ybuffer[])
{
	if (p0.x > p1.x)
	{
		std::swap(p0, p1);
	}
	for (int x = p0.x; x <= p1.x; ++x)
	{
		float t = (x - p0.x) / (float)(p1.x - p0.x);  // t 是  (0, 1)的一个插值的进度
		int y = p0.y * (1.0 - t) + p1.y * t;		// 计算出来p0 p1这条线上，t 对应的y
		if (ybuffer[x] < y)
		{
			ybuffer[x] = y;						// 更新ybuffer
			image.set(x, 0, color);
		}
	}
}

//void triangle(Vec3f* pts, float* zbuffer, TGAImage& image, TGAColor color)
//{
//	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
//	Vec2f bboxmax(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
//	Vec2f clamp(image.get_width() - 1, image.get_height() - 1);
//	for (int i = 0; i < 3; ++i)
//		for (int j = 0; j < 2; ++j)
//		{
//			bboxmin[j] = std::fmax(0.0f, std::fmin(bboxmin[j], pts[i][j]));
//			bboxmax[j] = std::fmin(clamp[j], std::fmax(bboxmax[j], pts[i][j]));
//		}
//
//	Vec3f P;
//	for (P.x = bboxmin.x; P.x <= bboxmax.x; ++P.x)
//	{
//		for (P.y = bboxmin.y; P.y <= bboxmax.y; ++P.y)
//		{
//			Vec3f bc_screen = barycentric(pts[0], pts[1], pts[2], P); // 原先这个 bc_screen 是用来判定是不是在三角形内的
//			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
//			P.z = 0;
//			for (int i = 0; i < 3; ++i)
//				P.z += pts[i][2] * bc_screen[i];   // 计算出来三角形重心的z 然后去和zbuffer比较
//			if (zbuffer[int(P.x + P.y * width)] < P.z)
//			{
//				zbuffer[int(P.x + P.y * width)] = P.z;
//				image.set(P.x, P.y, color);
//			}
//		}
//	}
//}

void triangle(Vec3i t0, Vec3i t1, Vec3i t2, Vec2i uv0, Vec2i uv1, Vec2i uv2, TGAImage& image, float intensity, int* zbuffer)
{
	if (t0.y == t1.y && t0.y == t2.y) return;
	if (t0.y > t1.y) { std::swap(t0, t1); std::swap(uv0, uv1); }
	if (t0.y > t2.y) { std::swap(t0, t2); std::swap(uv0, uv2); }
	if (t1.y > t2.y) { std::swap(t1, t2); std::swap(uv1, uv2); }

	int total_height = t2.y - t0.y;
	for (int i = 0; i < total_height; ++i)
	{
		bool second_half = i > t1.y - t0.y || t1.y == t0.y;
		int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
		float alpha = (float)i / total_height;
		float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height;
		Vec3i A = t0 + Vec3f(t2 - t0) * alpha;
		Vec3i B = second_half ? t1 + Vec3f(t2 - t1) * beta : t0 + Vec3f(t1 - t0) * beta;

		Vec2i uvA = uv0 + (uv2 - uv0) * alpha;
		Vec2i uvB = second_half ? uv1 + (uv2 - uv1) * beta : uv0 + (uv1 - uv0) * beta;

		if (A.x > B.x)
		{
			std::swap(A, B);
			std::swap(uvA, uvB);
		}

		for (int j = A.x; j <= B.x; ++j)
		{
			float phi = B.x == A.x ? 1.0 : (float)(j - A.x) / (float)(B.x - A.x);
			Vec3i P = Vec3f(A) + Vec3f(B - A) * phi;
			Vec2i uvP = uvA + (uvB - uvA) * phi;
			int idx = P.x + P.y * width;
			if (zbuffer[idx] < P.z)
			{
				zbuffer[idx] = P.z;
				TGAColor color = model->diffuse(uvP);
				image.set(P.x, P.y, TGAColor(color.r * intensity, color.g * intensity, color.b * intensity));
			}
		}
	}
}

Vec3f world2screen(Vec3f v) {
	return Vec3f(int((v.x + 1.) * width / 2. + .5), int((v.y + 1.) * height / 2. + .5), v.z);
}


void lookat(Vec3f eye, Vec3f center, Vec3f up)
{
	Vec3f z = (eye - center).normalize();  // 看图上的意思是，ce组成lookat向量,作为新的z' 轴
	Vec3f x = cross(up, z).normalize();		// u 叉乘 z'  得到 x'
	Vec3f y = cross(z, x).normalize();
	Matrix MinV = Matrix::identity();
	Matrix Tr = Matrix::identity();
	for (int i = 0; i < 3; ++i)
	{
		MinV[0][i] = x[i];
		MinV[1][i] = y[i];
		MinV[2][i] = z[i];
		Tr[i][3] = -center[i];
	}
	ModelView = MinV * Tr;
}

int main(int argc, char** argv)
{
	//TGAImage image(100, 100, TGAImage::RGB);
	//image.set(52, 41, red);
	////line(0, 0, 52, 41, image, white);
	//line(13, 20, 80, 40, image, white);
	//line(20, 13, 40, 80, image, red);
	//line(80, 40, 13, 20, image, red);
	//image.flip_vertically(); // 原点置于图片左下角
	//image.write_tga_file("output.tga");
	//return 0;
	//if (2 == argc)
	//{
	//	model = new Model(argv[1]);
	//}
	//else
	//{
	//	
	//}
	model = new Model("E:/TinyRenderer/TinyRenderer/african_head.obj");

	//printf("123");

	TGAImage image(width, height, TGAImage::RGB);
	//for (int i = 0; i < model->nfaces(); ++i) // 面
	//{
	//	std::vector<int> face = model->face(i);
	//	for (int j = 0; j < 3; j++)
	//	{
	//		Vec3f v0 = model->vert(face[j]);
	//		Vec3f v1 = model->vert(face[(j + 1) % 3]);
	//		int x0 = (v0.x + 1.0) * width / 2.0;
	//		int y0 = (v0.y + 1.0) * height / 2.0;
	//		int x1 = (v1.x + 1.0) * width / 2.0;
	//		int y1 = (v1.y + 1.0) * height / 2.0;
	//		line(x0, y0, x1, y1, image, white);
	//	}
	//}

	//Vec2i t0[3] = { Vec2i(10, 70),   Vec2i(50, 160),  Vec2i(70, 80) };
	//Vec2i t1[3] = { Vec2i(180, 50),  Vec2i(150, 1),   Vec2i(70, 180) };
	//Vec2i t2[3] = { Vec2i(180, 150), Vec2i(120, 160), Vec2i(130, 180) };
	//triangle(t0[0], t0[1], t0[2], image, red);
	//triangle(t1[0], t1[1], t1[2], image, white);
	//triangle(t2[0], t2[1], t2[2], image, green);

	/*Vec2i pts[3] = { Vec2i(10,10), Vec2i(100, 30), Vec2i(190, 160) };
	triangle(pts, image, red);*/

	/// 随机颜色的渲染
	/*for (int i = 0; i < model->nfaces(); i++) {
		std::vector<int> face = model->face(i);
		Vec2i screen_coords[3];
		for (int j = 0; j < 3; j++) {
			Vec3f world_coords = model->vert(face[j]);
			screen_coords[j] = Vec2i((world_coords.x + 1.) * width / 2., (world_coords.y + 1.) * height / 2.);
		}
		triangle(screen_coords[0], screen_coords[1], screen_coords[2], image, TGAColor(rand() % 255, rand() % 255, rand() % 255, 255));
	}*/


	//Vec3f light_dir(0, 0, -1);
	//for (int i = 0; i < model->nfaces(); ++i)
	//{
	//	std::vector<int> face = model->face(i);
	//	Vec2i screen_coords[3];
	//	Vec3f world_coords[3];
	//	for (int j = 0; j < 3; ++j)
	//	{
	//		Vec3f v = model->vert(face[j]);
	//		screen_coords[j] = Vec2i((v.x + 1.0) * width / 2.0, (v.y + 1.0) * height / 2.0);
	//		world_coords[j] = v;
	//	}
	//	Vec3f n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]); // ^符号被重载了，作为cross
	//	n.normalize();
	//	float intensity = n * light_dir;
	//	if (intensity > 0)
	//	{
	//		triangle(screen_coords[0], screen_coords[1], screen_coords[2], image,
	//			TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));

	//		// 经测试，包围盒这个算法比上面这个慢很多很多倍，不知道为啥
	//		/*Vec2i temp[3] = { screen_coords[0], screen_coords[1], screen_coords[2] };
	//		triangle(temp, image,
	//			TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));*/
	//	}
	//}

	// 这块儿的代码是用来做ybuffer的，就是在1维的一条线上展示一下ybuffer的原理
	//{
	//TGAImage image(width, 16, TGAImage::RGB);
	//int ybuffer[width];
	//for (int i = 0; i < width; ++i)
	//{
	//	ybuffer[i] = std::numeric_limits<int>::min();
	//}

	//// scene "2d mesh"
	//rasterize(Vec2i(20, 34), Vec2i(744, 400), image, red, ybuffer);
	//rasterize(Vec2i(120, 434), Vec2i(444, 400), image, green, ybuffer);
	//rasterize(Vec2i(330, 463), Vec2i(594, 200), image, blue, ybuffer);
	// screen line 
	//line(Vec2i(10, 10), Vec2i(790, 10), image, white);


//}
	//Vec3f light_dir(0, 0, -1);
	//int* zbuffer = new int[width * height];
	Matrix Projection = Matrix::identity(4);
	Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
	Projection[3][2] = -1.0f / camera.z;

	zbuffer = new int[width * height];
	for (int i = 0; i < width * height; i++)
		zbuffer[i] = -std::numeric_limits<int>::min();

	for (int i = 0; i < model->nfaces(); ++i)
	{
		std::vector<int> face = model->face(i);
		Vec3i screen_coords[3];
		Vec3f world_coords[3];
		for (int j = 0; j < 3; ++j)
		{
			//screen_coords[j] = world2screen(model->vert(face[j]));
			Vec3f v = model->vert(face[j]);
			screen_coords[j] = m2v(ViewPort * Projection * v2m(v));
			world_coords[j] = v;
		}

		Vec3f n = (model->vert(face[2]) - model->vert(face[0])) ^ (model->vert(face[1]) - model->vert(face[0]));
		n.normalize();
		float intensity = n * light_dir;
		if (intensity > 0)
		{
			Vec2i uv[3];
			for (int k = 0; k < 3; ++k)
			{
				uv[k] = model->uv(i, k);
			}
			//triangle(pts, zbuffer, image, TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));
			triangle(screen_coords[0], screen_coords[1], screen_coords[2], uv[0], uv[1], uv[2], image, intensity, zbuffer);
		}
	}

	image.flip_vertically();
	image.write_tga_file("output.tga");
	delete model;
	delete[] zbuffer;
	return 0;
}

