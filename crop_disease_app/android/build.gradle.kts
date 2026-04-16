allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
    
    // Using withPlugin ensures this runs even if the plugin was already evaluated
    pluginManager.withPlugin("com.android.library") {
        project.dependencies {
            add("implementation", "androidx.concurrent:concurrent-futures:1.2.0")
            add("compileOnly", "androidx.concurrent:concurrent-futures:1.2.0")
            add("implementation", "com.google.guava:guava:31.1-android")
            add("implementation", "org.jspecify:jspecify:1.0.0")
        }
    }
    pluginManager.withPlugin("com.android.application") {
        project.dependencies {
            add("implementation", "androidx.concurrent:concurrent-futures:1.2.0")
            add("compileOnly", "androidx.concurrent:concurrent-futures:1.2.0")
            add("implementation", "com.google.guava:guava:31.1-android")
            add("implementation", "org.jspecify:jspecify:1.0.0")
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
